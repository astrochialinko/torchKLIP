document.addEventListener("DOMContentLoaded", () => {
  const DATA_URL = "../_static/data/betaPic_PCA.json";
  const WIDTH = 600,
    HEIGHT = 600;
  const MARGIN = { top: 40, right: 80, bottom: 40, left: 40 };
  const innerW = WIDTH - MARGIN.left - MARGIN.right;
  const innerH = HEIGHT - MARGIN.top - MARGIN.bottom;

  // 1) create canvas
  const canvas = document.createElement("canvas");
  canvas.width = WIDTH;
  canvas.height = HEIGHT;
  document.getElementById("pca-container").appendChild(canvas);
  const ctx = canvas.getContext("2d");

  // grab tooltip div
  const tooltip = document.getElementById("pca-tooltip");

  // 2) slider & label
  const slider = document.getElementById("pca-slider");
  const label = document.getElementById("pca-label");
  let min = +slider.min,
    max = +slider.max;
  function updateLabel() {
    const v = +slider.value;
    const pct = ((v - min) / (max - min)) * 100;
    slider.style.setProperty("--pct", pct + "%");
    label.style.left = pct + "%";
    label.textContent = v;
  }
  slider.addEventListener("input", () => {
    updateLabel();
    drawFrame();
  });

  // 3) theme detection
  let isDark = document.documentElement.dataset.theme === "dark";
  new MutationObserver((muts) => {
    muts.forEach((m) => {
      if (m.attributeName === "data-theme") {
        isDark = document.documentElement.dataset.theme === "dark";
        drawFrame();
      }
    });
  }).observe(document.documentElement, { attributes: true });

  // 4) data vars
  let cube, rows, cols, pxW, pxH, mn, mx;

  // 5) load and init
  d3.json(DATA_URL).then((raw) => {
    cube = raw.PCA_k; // shape: [frames][rows][cols]
    const nFrames = cube.length;
    rows = cube[0].length;
    cols = cube[0][0].length;
    pxW = innerW / cols;
    pxH = innerH / rows;

    // slider range
    slider.min = 1;
    slider.max = nFrames;
    min = 1;
    max = nFrames;
    updateLabel();

    // global data range
    const flat = cube.flat(2);
    [mn, mx] = d3.extent(flat);

    // mouse → tooltip
    canvas.addEventListener("mousemove", (e) => {
      const rect = canvas.getBoundingClientRect();
      const mx0 = e.clientX - rect.left;
      const my0 = e.clientY - rect.top;
      const x = mx0 - MARGIN.left;
      const y = my0 - MARGIN.top;
      const j0 = Math.floor(x / pxW);
      const i0 = Math.floor(y / pxH);
      const idx = +slider.value - 1;
      if (i0 >= 0 && i0 < rows && j0 >= 0 && j0 < cols) {
        const v = cube[idx][i0][j0].toFixed(2);
        tooltip.style.left = `${mx0 + 8}px`;
        tooltip.style.top = `${my0 + 8}px`;
        tooltip.textContent = v;
        tooltip.style.opacity = 1;
      } else {
        tooltip.style.opacity = 0;
      }
    });
    canvas.addEventListener("mouseout", () => {
      tooltip.style.opacity = 0;
    });

    // initial draw
    drawFrame();
  });

  // 6) drawFrame: heatmap + axes + colorbar
  function drawFrame() {
    if (!cube) return;
    const idx = +slider.value - 1;
    const frame = cube[idx];
    ctx.clearRect(0, 0, WIDTH, HEIGHT);

    // heatmap
    ctx.save();
    ctx.translate(MARGIN.left, MARGIN.top);
    const colorScale = d3.scaleLinear().domain([mn, mx]).range([0, 1]);
    const interp = d3.interpolateViridis;
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        ctx.fillStyle = interp(colorScale(frame[i][j]));
        ctx.fillRect(j * pxW, i * pxH, pxW + 0.5, pxH + 0.5);
      }
    }
    ctx.restore();

    // axes styling
    ctx.strokeStyle = isDark ? "white" : "black";
    ctx.fillStyle = isDark ? "white" : "black";
    ctx.lineWidth = 1;
    ctx.font = "12px sans-serif";

    // X axis
    ctx.beginPath();
    ctx.moveTo(MARGIN.left, MARGIN.top + innerH);
    ctx.lineTo(MARGIN.left + innerW, MARGIN.top + innerH);
    ctx.stroke();
    ctx.textAlign = "center";
    for (let t = 0; t <= 5; t++) {
      const x = MARGIN.left + innerW * (t / 5);
      const y = MARGIN.top + innerH;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x, y + 6);
      ctx.stroke();
      ctx.fillText(Math.round(cols * (t / 5)), x, y + 18);
    }

    // Y axis
    ctx.beginPath();
    ctx.moveTo(MARGIN.left, MARGIN.top);
    ctx.lineTo(MARGIN.left, MARGIN.top + innerH);
    ctx.stroke();
    ctx.textAlign = "right";
    for (let t = 0; t <= 5; t++) {
      const y = MARGIN.top + innerH * (1 - t / 5);
      ctx.beginPath();
      ctx.moveTo(MARGIN.left, y);
      ctx.lineTo(MARGIN.left - 6, y);
      ctx.stroke();
      ctx.fillText(Math.round(rows * (t / 5)), MARGIN.left - 8, y + 4);
    }

    let dataMn = -0.1;
    let dataMx = 0.1;

    // Draw colorbar (not affected by zoom)
    const barX = MARGIN.left + innerW + 20;
    const barY = MARGIN.top;
    const grad = ctx.createLinearGradient(0, barY + innerH, 0, barY);
    for (let t = 0; t <= 1; t += 0.1) {
      grad.addColorStop(t, interp(t));
    }
    ctx.fillStyle = grad;
    ctx.fillRect(barX, barY, 20, innerH);

    // Colorbar ticks with improved precision
    ctx.fillStyle = isDark ? "white" : "black";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "left";
    for (let k = 0; k <= 6; k++) {
      const y = barY + (innerH / 6) * k;
      const v = dataMx - (dataMx - dataMn) * (k / 6);

      // Format with appropriate precision based on data range
      const range = Math.abs(dataMx - dataMn);
      let formatted;
      if (range < 0.1) {
        formatted = v.toFixed(3); // Use more decimal places for small ranges
      } else if (range < 1) {
        formatted = v.toFixed(2);
      } else if (range < 10) {
        formatted = v.toFixed(1);
      } else {
        formatted = v.toFixed(0);
      }

      ctx.fillText(formatted, barX + 26, y + 4);
    }
  }
});

document.addEventListener("DOMContentLoaded", () => {
  const DATA_URL = "../_static/data/betaPic.json";
  const WIDTH = 600,
    HEIGHT = 600;
  const MARGIN = { top: 40, right: 80, bottom: 40, left: 40 };
  const innerW = WIDTH - MARGIN.left - MARGIN.right;
  const innerH = HEIGHT - MARGIN.top - MARGIN.bottom;

  // 1) Create canvas
  const container = document.getElementById("plot-container");
  const canvas = document.createElement("canvas");
  canvas.width = WIDTH + MARGIN.right;
  canvas.height = HEIGHT + MARGIN.bottom;
  container.appendChild(canvas);
  const ctx = canvas.getContext("2d");

  // 2) Zoom state (no pan)
  let scale = 1,
    offsetX = 0,
    offsetY = 0;
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const delta = -e.deltaY * 0.001;
    const newScale = Math.min(10, Math.max(1, scale + delta));
    const factor = newScale / scale;
    // zoom about plot center (fixed axes)
    const cx = MARGIN.left + innerW / 2,
      cy = MARGIN.top + innerH / 2;
    offsetX = cx - factor * (cx - offsetX);
    offsetY = cy - factor * (cy - offsetY);
    scale = newScale;
    redraw();
  });

  let isDragging = false,
    lastX,
    lastY;

  canvas.addEventListener("mousedown", (e) => {
    isDragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
  });

  window.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    // compute delta
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    // update pan offsets
    offsetX += dx;
    offsetY += dy;
    redraw();
  });

  window.addEventListener("mouseup", () => {
    isDragging = false;
  });

  // 3) Controls + floating label
  const slider = document.getElementById("param-slider");
  const bubble = document.getElementById("param-label");
  const cbLog = document.getElementById("log-scale");
  const cbIWA = document.getElementById("show-iwa");
  const cbCenter = document.getElementById("show-center");
  let min = +slider.min,
    max = +slider.max;

  slider.addEventListener("input", () => {
    update();
    redraw();
  });
  [cbLog, cbIWA, cbCenter].forEach((el) =>
    el.addEventListener("input", () => {
      redraw();
    })
  );

  function update() {
    const val = +slider.value;
    const pct = ((val - min) / (max - min)) * 100;
    slider.style.setProperty("--pct", pct + "%");
    bubble.style.left = pct + "%";
    bubble.textContent = val;
  }

  let isDark = document.documentElement.dataset.theme === "dark";
  const root = document.documentElement;
  const mo = new MutationObserver((muts) => {
    muts.forEach((m) => {
      if (m.type === "attributes" && m.attributeName === "data-theme") {
        // theme just changed
        isDark = document.documentElement.dataset.theme === "dark";
        redraw();
      }
    });
  });
  mo.observe(root, { attributes: true });

  // 4) Data load & drawing
  let redraw = () => {};
  d3.json(DATA_URL).then((raw) => {
    const cube = raw.data_init;
    const nFrames = cube.length;
    const rows = cube[0].length;
    const cols = cube[0][0].length;
    const pxW = innerW / cols,
      pxH = innerH / rows;

    // set slider range & init label
    slider.min = 1;
    slider.max = nFrames;
    min = 1;
    max = nFrames;
    update();

    // real redraw implementation
    redraw = () => {
      // clear entire canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // ─── Clip to plot area ─────────────────────────────────────────────────────
      ctx.save();
      ctx.beginPath();
      ctx.rect(MARGIN.left, MARGIN.top, innerW, innerH);
      ctx.clip();

      // ─── Zoomed heatmap ─────────────────────────────────────────────────────
      ctx.save();
      ctx.translate(offsetX, offsetY);
      ctx.scale(scale, scale);
      ctx.translate(MARGIN.left, MARGIN.top);

      // frame selection & color setup
      const idx = +slider.value - 1;
      const frame = cube[idx];
      const domain = cbLog.checked ? [1e2, 1e4] : [0, 2500];
      const colorScale = cbLog.checked
        ? d3.scaleLog().domain(domain).range([0, 1])
        : d3.scaleLinear().domain(domain).range([0, 1]);
      const interp = d3.interpolateViridis;

      // draw pixels
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          ctx.fillStyle = interp(colorScale(frame[i][j]));
          ctx.fillRect(j * pxW, i * pxH, pxW + 0.5, pxH + 0.5);
        }
      }

      // IWA circle
      if (cbIWA.checked) {
        ctx.strokeStyle = "white";
        ctx.setLineDash([4, 4]);
        const cx = (cols / 2) * pxW + 7,
          cy = (rows / 2) * pxH + 7;
        ctx.beginPath();
        ctx.arc(cx, cy, 4.065 * pxW, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // center star
      if (cbCenter.checked) {
        ctx.fillStyle = "black";
        const cx = (cols / 2) * pxW + 7,
          cy = (rows / 2) * pxH + 7;
        ctx.beginPath();
        for (let k = 0; k < 5; k++) {
          const a = (2 * Math.PI * k) / 5 - Math.PI / 2;
          ctx.lineTo(cx + Math.cos(a) * 6, cy + Math.sin(a) * 6);
          const a2 = a + Math.PI / 5;
          ctx.lineTo(cx + Math.cos(a2) * 2, cy + Math.sin(a2) * 2);
        }
        ctx.closePath();
        ctx.fill();
      }

      ctx.restore(); // end zoom transform
      ctx.restore(); // end clipping

      // ─── Static colorbar ─────────────────────────────────────────────────────
      const barX = MARGIN.left + innerW + 20,
        barY = MARGIN.top;
      const grad = ctx.createLinearGradient(0, barY + innerH, 0, barY);
      for (let t = 0; t <= 1; t += 0.1) {
        grad.addColorStop(t, d3.interpolateViridis(t));
      }
      ctx.fillStyle = grad;
      ctx.fillRect(barX, barY, 20, innerH);

      ctx.save();
      ctx.fillStyle = isDark ? "white" : "black";
      ctx.font = "12px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Counts", barX + 10, barY - 6);
      ctx.restore();

      ctx.fillStyle = isDark ? "white" : "black";
      ctx.font = "10px sans-serif";
      ctx.textAlign = "left";
      for (let i = 0; i <= 6; i++) {
        const y = barY + (innerH / 6) * i;
        const v = domain[1] - (domain[1] - domain[0]) * (i / 6);
        ctx.fillText(v.toFixed(0), barX + 24, y + 4);
      }

      // ─── Static axes & ticks ─────────────────────────────────────────────────
      ctx.strokeStyle = isDark ? "white" : "black";
      ctx.lineWidth = 1;
      ctx.fillStyle = isDark ? "white" : "black";

      // X axis
      ctx.beginPath();
      ctx.moveTo(MARGIN.left, MARGIN.top + innerH);
      ctx.lineTo(MARGIN.left + innerW, MARGIN.top + innerH);
      ctx.stroke();
      ctx.textAlign = "center";
      for (let i = 0; i <= 5; i++) {
        const x = MARGIN.left + innerW * (i / 5);
        ctx.beginPath();
        ctx.moveTo(x, MARGIN.top + innerH);
        ctx.lineTo(x, MARGIN.top + innerH + 6);
        ctx.stroke();
        ctx.fillText(Math.round(cols * (i / 5)), x, MARGIN.top + innerH + 16);
      }

      // Y axis
      ctx.beginPath();
      ctx.moveTo(MARGIN.left, MARGIN.top);
      ctx.lineTo(MARGIN.left, MARGIN.top + innerH);
      ctx.stroke();
      ctx.textAlign = "right";
      for (let i = 0; i <= 5; i++) {
        const y = MARGIN.top + innerH * (1 - i / 5);
        ctx.beginPath();
        ctx.moveTo(MARGIN.left, y);
        ctx.lineTo(MARGIN.left - 6, y);
        ctx.stroke();
        ctx.fillText(Math.round(rows * (i / 5)), MARGIN.left - 8, y + 4);
      }
    };

    redraw();

    const tooltip = document.getElementById("pixel-tooltip");

    // on mouse move, compute data‐index and show tooltip
    canvas.addEventListener("mousemove", (e) => {
      // get mouse coords relative to canvas
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      // transform back through pan/zoom
      const x = (mx - offsetX - MARGIN.left) / scale;
      const y = (my - offsetY - MARGIN.top) / scale;

      // convert to grid indices
      const j = Math.floor(x / pxW);
      const i = Math.floor(y / pxH);

      // check bounds
      if (i >= 0 && i < rows && j >= 0 && j < cols) {
        const idx = +slider.value - 1;
        const val = cube[idx][i][j].toFixed(2);

        // position and show
        tooltip.style.left = `${mx + 10}px`;
        tooltip.style.top = `${my + 10}px`;
        tooltip.textContent = val;
        tooltip.style.opacity = 1;
      } else {
        tooltip.style.opacity = 0;
      }
    });

    // hide when leaving canvas
    canvas.addEventListener("mouseout", () => {
      tooltip.style.opacity = 0;
    });
  });
});

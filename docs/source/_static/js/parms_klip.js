document.addEventListener("DOMContentLoaded", () => {
  const DATA_URL = "../_static/data/betaPic.json";

  // Smaller width for each plot
  const WIDTH = 450,
    HEIGHT = 450;
  const MARGIN = { top: 40, right: 80, bottom: 40, left: 40 };
  const innerW = WIDTH - MARGIN.left - MARGIN.right;
  const innerH = HEIGHT - MARGIN.top - MARGIN.bottom;
  let redrawK = () => {};
  let drawSNR = () => {};

  // ─── 1) create canvases & contexts ───────────────────────────────────────────
  const createCanvas = (id) => {
    const c = document.createElement("canvas");
    c.width = WIDTH;
    c.height = HEIGHT;
    document.querySelector(id).appendChild(c);
    return { canvas: c, ctx: c.getContext("2d") };
  };
  const { canvas: canvasK, ctx: ctxK } = createCanvas("#klip-container");
  const { canvas: canvasS, ctx: ctxS } = createCanvas("#snr-container");

  // ─── 2) zoom & pan state for KLIP plot only ─────────────────────────────────
  let scale = 1,
    offsetX = 0,
    offsetY = 0;
  canvasK.addEventListener("wheel", (e) => {
    e.preventDefault();
    const delta = -e.deltaY * 0.001;
    const newScale = Math.min(10, Math.max(1, scale + delta));
    const factor = newScale / scale;
    // zoom about KLIP plot center
    const cx = MARGIN.left + innerW / 2;
    const cy = MARGIN.top + innerH / 2;
    offsetX = cx - factor * (cx - offsetX);
    offsetY = cy - factor * (cy - offsetY);
    scale = newScale;
    redrawK();
  });
  let isDragging = false,
    dragStartX,
    dragStartY;
  canvasK.addEventListener("mousedown", (e) => {
    isDragging = true;
    dragStartX = e.clientX - offsetX;
    dragStartY = e.clientY - offsetY;
  });
  window.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    offsetX = e.clientX - dragStartX;
    offsetY = e.clientY - dragStartY;
    redrawK();
  });
  window.addEventListener("mouseup", () => {
    isDragging = false;
  });

  // ─── 3) slider + theme controls ─────────────────────────────────────────────
  const slider = document.getElementById("klip-slider");
  const bubble = document.getElementById("klip-label");
  let min = +slider.min,
    max = +slider.max;
  slider.addEventListener("input", () => {
    redrawK();
    drawSNR();
    updateSliderLabel();
  });
  function updateSliderLabel() {
    const val = +slider.value;
    const pct = ((val - min) / (max - min)) * 100;
    slider.style.setProperty("--pct", pct + "%");
    bubble.style.left = pct + "%";
    bubble.textContent = val;
  }
  updateSliderLabel();

  let isDark = document.documentElement.dataset.theme === "dark";
  new MutationObserver((muts) => {
    muts.forEach((m) => {
      if (m.type === "attributes" && m.attributeName === "data-theme") {
        isDark = document.documentElement.dataset.theme === "dark";
        redrawK();
        drawSNR();
      }
    });
  }).observe(document.documentElement, { attributes: true });

  // ─── 4) load data & implement redraws ────────────────────────────────────────
  d3.json(DATA_URL).then((raw) => {
    const cube = raw.data_klip_k;
    const kList = raw.k_list;
    const snrList = raw.snr_list;
    const rows = cube[0].length,
      cols = cube[0][0].length;
    const pxW = innerW / cols,
      pxH = innerH / rows;

    // D3 scales (reuse)
    const xScaleK = d3.scaleLinear().domain([0, cols]).range([0, innerW]);
    const yScaleK = d3.scaleLinear().domain([0, rows]).range([innerH, 0]);
    const xScaleS = d3
      .scaleLinear()
      .domain(d3.extent(kList))
      .range([0, innerW]);
    const yScaleS = d3
      .scaleLinear()
      .domain(d3.extent(snrList))
      .nice()
      .range([innerH, 0]);

    // ─── redrawK: KLIP heatmap with zoom, axes, colorbar ──────────────────────
    redrawK = () => {
      // clear
      ctxK.clearRect(0, 0, WIDTH, HEIGHT);

      // clip to plot area
      ctxK.save();
      ctxK.beginPath();
      ctxK.rect(MARGIN.left, MARGIN.top, innerW, innerH);
      ctxK.clip();

      // apply zoom+pan, then margin offset
      ctxK.save();
      ctxK.translate(offsetX, offsetY);
      ctxK.scale(scale, scale);
      ctxK.translate(MARGIN.left, MARGIN.top);

      // draw pixels
      const idx = +slider.value - 1;
      const frame2D = cube[idx];
      const flat = frame2D.flat();
      const [mn, mx] = d3.extent(flat);
      const colorK = d3.scaleSequential(d3.interpolateViridis).domain([mn, mx]);

      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          ctxK.fillStyle = colorK(frame2D[i][j]);
          ctxK.fillRect(j * pxW, i * pxH, pxW + 0.5, pxH + 0.5);
        }
      }

      ctxK.restore(); // end zoom+pan
      ctxK.restore(); // end clip

      // draw static colorbar
      const barX = MARGIN.left + innerW + 20;
      const barY = MARGIN.top;
      const grad = ctxK.createLinearGradient(0, barY + innerH, 0, barY);
      d3.range(0, 1.01, 0.1).forEach((t) => {
        grad.addColorStop(t, d3.interpolateViridis(t));
      });
      ctxK.fillStyle = grad;
      ctxK.fillRect(barX, barY, 20, innerH);

      // colorbar label
      ctxK.fillStyle = isDark ? "white" : "black";
      ctxK.font = "12px sans-serif";
      ctxK.textAlign = "center";
      ctxK.fillText("Counts", barX + 10, barY - 6);

      // colorbar ticks
      ctxK.fillStyle = isDark ? "white" : "black";
      ctxK.font = "10px sans-serif";
      ctxK.textAlign = "left";
      for (let i = 0; i <= 6; i++) {
        const y = barY + (innerH / 6) * i;
        const v = mx - (mx - mn) * (i / 6);
        ctxK.fillText(v.toFixed(2), barX + 24, y + 4);
      }

      // draw static axes
      ctxK.strokeStyle = isDark ? "white" : "black";
      ctxK.lineWidth = 1;
      ctxK.fillStyle = isDark ? "white" : "black";

      // X axis
      ctxK.beginPath();
      ctxK.moveTo(MARGIN.left, MARGIN.top + innerH);
      ctxK.lineTo(MARGIN.left + innerW, MARGIN.top + innerH);
      ctxK.stroke();
      ctxK.textAlign = "center";
      for (let t = 0; t <= 5; t++) {
        const x = MARGIN.left + innerW * (t / 5);
        ctxK.beginPath();
        ctxK.moveTo(x, MARGIN.top + innerH);
        ctxK.lineTo(x, MARGIN.top + innerH + 6);
        ctxK.stroke();
        ctxK.fillText(Math.round(cols * (t / 5)), x, MARGIN.top + innerH + 16);
      }

      // Y axis
      ctxK.beginPath();
      ctxK.moveTo(MARGIN.left, MARGIN.top);
      ctxK.lineTo(MARGIN.left, MARGIN.top + innerH);
      ctxK.stroke();
      ctxK.textAlign = "right";
      for (let t = 0; t <= 5; t++) {
        const y = MARGIN.top + innerH * (1 - t / 5);
        ctxK.beginPath();
        ctxK.moveTo(MARGIN.left, y);
        ctxK.lineTo(MARGIN.left - 6, y);
        ctxK.stroke();
        ctxK.fillText(Math.round(rows * (t / 5)), MARGIN.left - 8, y + 4);
      }
    };

    // ─── drawSNR: static SNR plot ───────────────────────────────────────────────
    drawSNR = () => {
      const idx = +slider.value - 1;
      ctxS.clearRect(0, 0, WIDTH, HEIGHT);
      ctxS.save();
      ctxS.translate(MARGIN.left, MARGIN.top);

      // line
      ctxS.beginPath();
      ctxS.moveTo(xScaleS(kList[0]), yScaleS(snrList[0]));
      snrList.forEach((d, i) => {
        ctxS.lineTo(xScaleS(kList[i]), yScaleS(d));
      });
      ctxS.strokeStyle = "#66c2a5";
      ctxS.lineWidth = 2;
      ctxS.stroke();

      // mover dot
      ctxS.beginPath();
      const cx = xScaleS(kList[idx]),
        cy = yScaleS(snrList[idx]);
      ctxS.arc(cx, cy, 6, 0, 2 * Math.PI);
      ctxS.fillStyle = "#fdb462";
      ctxS.fill();

      // axes
      ctxS.strokeStyle = isDark ? "white" : "black";
      ctxS.lineWidth = 1;
      ctxS.fillStyle = isDark ? "white" : "black";
      ctxS.font = "12px sans-serif";

      // X axis
      ctxS.beginPath();
      ctxS.moveTo(0, innerH);
      ctxS.lineTo(innerW, innerH);
      ctxS.stroke();
      ctxS.textAlign = "center";
      xScaleS.ticks(6).forEach((v) => {
        const x = xScaleS(v);
        ctxS.beginPath();
        ctxS.moveTo(x, innerH);
        ctxS.lineTo(x, innerH + 6);
        ctxS.stroke();
        ctxS.fillText(v, x, innerH + 20);
      });

      // Y axis
      ctxS.beginPath();
      ctxS.moveTo(0, 0);
      ctxS.lineTo(0, innerH);
      ctxS.stroke();
      ctxS.textAlign = "right";
      yScaleS.ticks(6).forEach((v) => {
        const y = yScaleS(v);
        ctxS.beginPath();
        ctxS.moveTo(0, y);
        ctxS.lineTo(-6, y);
        ctxS.stroke();
        ctxS.fillText(v, -8, y + 4);
      });

      // labels
      ctxS.textAlign = "center";
      ctxS.fillText("K Frames", innerW / 2, innerH + 40);
      ctxS.save();
      ctxS.translate(-30, innerH / 2);
      ctxS.rotate(-Math.PI / 2);
      ctxS.fillText("SNR", 0, 0);
      ctxS.restore();

      ctxS.restore();
    };

    // initial draw
    redrawK();
    drawSNR();

    // ─── tooltip on KLIP hover (no zoom adjustment needed here) ─────────────
    const tooltip = document.getElementById("klip-tooltip");
    canvasK.addEventListener("mousemove", (e) => {
      const rect = canvasK.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const x = mx - MARGIN.left;
      const y = my - MARGIN.top;
      const j = Math.floor(x / pxW);
      const i = Math.floor(y / pxH);
      if (i >= 0 && i < rows && j >= 0 && j < cols) {
        const idx = +slider.value - 1;
        const val = cube[idx][i][j].toFixed(2);
        tooltip.style.left = `${mx + 8}px`;
        tooltip.style.top = `${my + 8}px`;
        tooltip.textContent = val;
        tooltip.style.opacity = 1;
      } else {
        tooltip.style.opacity = 0;
      }
    });
    canvasK.addEventListener("mouseout", () => {
      tooltip.style.opacity = 0;
    });
  });
});

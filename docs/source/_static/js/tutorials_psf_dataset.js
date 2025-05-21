// improved_psf_viewer.js

document.addEventListener("DOMContentLoaded", () => {
  const URL = "../_static/data/betaPic_PSF.json";
  // Reduce canvas size for smaller figures
  const CW = 200,
    CH = 200;
  const Kvals = [5, 10, 20, 30, 40, 50, 60];

  // ── DOM refs ───────────────────────────────────────────────────────────────
  const sK = document.getElementById("pcsf-k-slider");
  const sT = document.getElementById("pcsf-t-slider");
  const lK = document.getElementById("pcsf-k-label");
  const lT = document.getElementById("pcsf-t-label");
  const cO = document.getElementById("pcsf-original");
  const cP = document.getElementById("pcsf-psf");
  const cR = document.getElementById("pcsf-residual");
  const ctxO = cO.getContext("2d");
  const ctxP = cP.getContext("2d");
  const ctxR = cR.getContext("2d");
  const tooltip = document.getElementById("pcsf-tooltip");
  const plotsContainer = document.getElementById("pcsf-plots");

  // Set canvas sizes to smaller dimensions
  cO.width = CW;
  cO.height = CH;
  cP.width = CW;
  cP.height = CH;
  cR.width = CW;
  cR.height = CH;

  // ── create two colorbar canvases ───────────────────────────────────────────
  const barO = document.createElement("canvas");
  barO.width = 20;
  barO.height = CH;
  barO.style.marginLeft = "8px";
  cO.parentNode.appendChild(barO);
  const ctxBO = barO.getContext("2d");

  const barP = document.createElement("canvas");
  barP.width = 20;
  barP.height = CH;
  barP.style.marginLeft = "8px";
  cP.parentNode.appendChild(barP);
  const ctxBP = barP.getContext("2d");

  const barR = document.createElement("canvas");
  barR.width = 20;
  barR.height = CH;
  barR.style.marginLeft = "8px";
  cR.parentNode.appendChild(barR);
  const ctxBR = barR.getContext("2d");

  let dataCube, psfCube, resCube;
  let domO, domP, domR;
  let rows, cols, pw, ph;

  // ── slider labels ─────────────────────────────────────────────────────────
  function updateLabels() {
    lK.textContent = Kvals[+sK.value - 1];
    lT.textContent = Kvals[+sT.value - 1];
  }
  [sK, sT].forEach((sl) =>
    sl.addEventListener("input", () => {
      updateLabels();
      drawAll();
    })
  );

  // ── load JSON ──────────────────────────────────────────────────────────────
  d3.json(URL).then((raw) => {
    dataCube = raw.data_t; // [T][N][N]
    psfCube = raw.pdf_k_t; // [T][K][N][N]
    resCube = raw.residuals_k_t; // [T][K][N][N]

    rows = dataCube[0].length;
    cols = dataCube[0][0].length;
    pw = CW / cols;
    ph = CH / rows;

    domO = d3.extent(dataCube.flat(2));
    domP = d3.extent(psfCube.flat(3));
    domR = d3.extent(resCube.flat(3));

    updateLabels();
    drawAll();
  });

  // ── draw one 2D frame ──────────────────────────────────────────────────────
  function drawImage(ctx, frame2d, domain) {
    const [mn, mx] = domain;
    const scale = d3.scaleLinear().domain([mn, mx]).range([0, 1]);
    const interp = d3.interpolateViridis;
    ctx.clearRect(0, 0, CW, CH);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        ctx.fillStyle = interp(scale(frame2d[i][j]));
        ctx.fillRect(j * pw, i * ph, pw, ph);
      }
    }
  }

  // ── draw a vertical Viridis colorbar ─────────────────────────────────────
  function drawColorbar(ctx, domain) {
    const w = ctx.canvas.width;
    const h = ctx.canvas.height;

    // 1) gradient
    const grad = ctx.createLinearGradient(0, h, 0, 0);
    for (let t = 0; t <= 1; t += 0.1) {
      grad.addColorStop(t, d3.interpolateViridis(t));
    }
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);

    // 2) ticks & labels - increased number of ticks
    const [mn, mx] = domain;
    const n = 6; // Number of ticks
    ctx.strokeStyle = "#fff";
    ctx.fillStyle = "#fff";
    ctx.font = "9px sans-serif"; // Smaller font for better fit
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";

    for (let i = 0; i <= n; i++) {
      const y = (h * i) / n;
      const v = mx - (mx - mn) * (i / n);
      // tick mark
      ctx.beginPath();
      ctx.moveTo(w - 4, y);
      ctx.lineTo(w, y);
      ctx.stroke();
      // text with 2 decimal places
      ctx.fillText(v.toFixed(2), w + 2, y);
    }

    // 3) "Value" label above
    ctx.fillStyle = "#fff";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText("Value", w / 2, -4);
  }

  // ── redraw everything ──────────────────────────────────────────────────────
  function drawAll() {
    if (!dataCube) return;
    const t = +sT.value - 1;
    const k = +sK.value - 1;

    // heatmaps
    drawImage(ctxO, dataCube[t], domO);
    drawImage(ctxP, psfCube[k][t], domP);
    drawImage(ctxR, resCube[k][t], domR);

    // colorbars for all three plots
    drawColorbar(ctxBO, domO);
    drawColorbar(ctxBP, domP);
    drawColorbar(ctxBR, domR);
  }

  // ── tooltips with value display ──────────────────────────────────────────
  [cO, cP, cR].forEach((canvasEl) => {
    canvasEl.addEventListener("mousemove", (e) => {
      const rCan = canvasEl.getBoundingClientRect();
      const rAll = plotsContainer.getBoundingClientRect();
      const mx = e.clientX - rCan.left,
        my = e.clientY - rCan.top;
      const i = Math.floor(my / ph),
        j = Math.floor(mx / pw);
      if (i < 0 || i >= rows || j < 0 || j >= cols) {
        tooltip.style.opacity = 0;
        return;
      }
      const t = +sT.value - 1,
        k = +sK.value - 1;
      let frame2d = dataCube[t];
      if (canvasEl === cP) frame2d = psfCube[t][k];
      if (canvasEl === cR) frame2d = resCube[t][k];
      const val = frame2d[i][j].toFixed(2);

      // Display value directly on canvas
      const valueOverlay = document.getElementById("pcsf-value-overlay");
      if (valueOverlay) {
        valueOverlay.textContent = val;
        valueOverlay.style.display = "block";

        // Position in center of image for reference value
        if (i === Math.floor(rows / 2) && j === Math.floor(cols / 2)) {
          const ctx = canvasEl.getContext("2d");
          ctx.fillStyle = "rgba(0,0,0,0.7)";
          ctx.fillRect(CW / 2 - 25, CH / 2 - 10, 50, 20);
          ctx.fillStyle = "#fff";
          ctx.font = "12px sans-serif";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(val, CW / 2, CH / 2);
        }
      }

      const px = e.clientX - rAll.left + 8;
      const py = e.clientY - rAll.top + 8;
      tooltip.textContent = val;
      tooltip.style.left = px + "px";
      tooltip.style.top = py + "px";
      tooltip.style.opacity = 1;
    });
    canvasEl.addEventListener("mouseout", () => {
      tooltip.style.opacity = 0;
    });
  });
});

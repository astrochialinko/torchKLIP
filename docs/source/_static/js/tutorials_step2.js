document.addEventListener("DOMContentLoaded", function () {
  // Get the target div
  const targetDiv = document.getElementById("step2");

  if (!targetDiv) {
    console.error("Target div with id 'step2' not found");
    return;
  }

  // Current step state
  let currentStep = 1;

  // Create the main container
  const mainContainer = document.createElement("div");
  mainContainer.className =
    "flex flex-col items-center w-full p-6 bg-gray-50 rounded-lg";

  // Add title
  const title = document.createElement("h2");
  title.className = "text-2xl font-bold mb-6 text-blue-700";
  title.textContent = "Computing KL Basis in KLIP Algorithm";
  mainContainer.appendChild(title);

  // Create navigation container
  const navContainer = document.createElement("div");
  navContainer.className = "flex items-center justify-between w-full mb-8";

  // Previous button
  const prevButton = document.createElement("button");
  prevButton.className = "px-4 py-2 rounded bg-gray-300";
  prevButton.textContent = "Previous";
  prevButton.disabled = true;

  // Step indicator
  const stepIndicator = document.createElement("div");
  stepIndicator.className = "text-lg font-semibold";
  stepIndicator.textContent = "Step 1 of 4";

  // Next button
  const nextButton = document.createElement("button");
  nextButton.className =
    "px-4 py-2 rounded bg-blue-600 text-white hover:bg-blue-700";
  nextButton.textContent = "Next";

  // Add navigation elements to container
  navContainer.appendChild(prevButton);
  navContainer.appendChild(stepIndicator);
  navContainer.appendChild(nextButton);
  mainContainer.appendChild(navContainer);

  // Create content container
  const contentContainer = document.createElement("div");
  contentContainer.className = "w-full bg-white p-6 rounded-lg shadow-md";
  mainContainer.appendChild(contentContainer);

  // Add the main container to the target div
  targetDiv.appendChild(mainContainer);

  // Event listeners for navigation
  prevButton.addEventListener("click", function () {
    if (currentStep > 1) {
      currentStep--;
      updateUI();
    }
  });

  nextButton.addEventListener("click", function () {
    if (currentStep < 4) {
      currentStep++;
      updateUI();
    }
  });

  // Function to update the UI based on current step
  function updateUI() {
    // Update step indicator
    stepIndicator.textContent = `Step ${currentStep} of 4`;

    // Update button states
    prevButton.disabled = currentStep === 1;
    prevButton.className = `px-4 py-2 rounded ${
      currentStep === 1
        ? "bg-gray-300"
        : "bg-blue-600 text-white hover:bg-blue-700"
    }`;

    nextButton.disabled = currentStep === 4;
    nextButton.className = `px-4 py-2 rounded ${
      currentStep === 4
        ? "bg-gray-300"
        : "bg-blue-600 text-white hover:bg-blue-700"
    }`;

    // Clear content container
    contentContainer.innerHTML = "";

    // Add appropriate content based on step
    switch (currentStep) {
      case 1:
        renderStep1();
        break;
      case 2:
        renderStep2();
        break;
      case 3:
        renderStep3();
        break;
      case 4:
        renderStep4();
        break;
    }
  }

  // Render functions for each step
  function renderStep1() {
    const stepDiv = document.createElement("div");
    stepDiv.className = "flex flex-col items-center";

    const heading = document.createElement("h3");
    heading.className = "text-xl font-semibold mb-4";
    heading.textContent = "Reference PSF Library Collection";
    stepDiv.appendChild(heading);

    const gridContainer = document.createElement("div");
    gridContainer.className = "grid grid-cols-3 gap-4 mb-6";

    // Create PSF library circles
    for (let i = 1; i <= 6; i++) {
      const psfContainer = document.createElement("div");
      psfContainer.className =
        "bg-gray-200 rounded-lg p-2 flex items-center justify-center";

      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("width", "100");
      svg.setAttribute("height", "100");
      svg.setAttribute("viewBox", "0 0 100 100");

      // Main circles
      const circle1 = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "circle"
      );
      circle1.setAttribute("cx", "50");
      circle1.setAttribute("cy", "50");
      circle1.setAttribute("r", 30 + i * 2);
      circle1.setAttribute("fill", "rgba(30, 64, 175, 0.1)");
      svg.appendChild(circle1);

      const circle2 = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "circle"
      );
      circle2.setAttribute("cx", "50");
      circle2.setAttribute("cy", "50");
      circle2.setAttribute("r", 20 + i);
      circle2.setAttribute("fill", "rgba(30, 64, 175, 0.2)");
      svg.appendChild(circle2);

      const circle3 = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "circle"
      );
      circle3.setAttribute("cx", "50");
      circle3.setAttribute("cy", "50");
      circle3.setAttribute("r", 10 + i / 2);
      circle3.setAttribute("fill", "rgba(30, 64, 175, 0.3)");
      svg.appendChild(circle3);

      // Add random speckles
      for (let j = 0; j < 15; j++) {
        const angle = Math.random() * Math.PI * 2;
        const radius = 10 + Math.random() * 30;
        const x = 50 + Math.cos(angle) * radius;
        const y = 50 + Math.sin(angle) * radius;

        const speckle = document.createElementNS(
          "http://www.w3.org/2000/svg",
          "circle"
        );
        speckle.setAttribute("cx", x);
        speckle.setAttribute("cy", y);
        speckle.setAttribute("r", 1 + Math.random() * 2);
        speckle.setAttribute(
          "fill",
          `rgba(30, 64, 175, ${0.3 + Math.random() * 0.5})`
        );
        svg.appendChild(speckle);
      }

      psfContainer.appendChild(svg);
      gridContainer.appendChild(psfContainer);
    }

    stepDiv.appendChild(gridContainer);

    // Description
    const description = document.createElement("p");
    description.className = "text-gray-700 text-center max-w-2xl";
    description.textContent =
      "The algorithm starts with a library of reference PSF frames that capture the speckle noise patterns and PSF structure. These frames are carefully selected to have sufficient sky rotation to avoid self-subtraction of potential companions.";
    stepDiv.appendChild(description);

    contentContainer.appendChild(stepDiv);
  }

  function renderStep2() {
    const stepDiv = document.createElement("div");
    stepDiv.className = "flex flex-col items-center";

    const heading = document.createElement("h3");
    heading.className = "text-xl font-semibold mb-4";
    heading.textContent = "Preprocessing & Matrix Formation";
    stepDiv.appendChild(heading);

    // Create visualization container
    const vizContainer = document.createElement("div");
    vizContainer.className = "flex items-center justify-center space-x-8 mb-6";

    // Reference frames
    const framesContainer = document.createElement("div");
    framesContainer.className = "flex flex-col items-center";

    const framesGrid = document.createElement("div");
    framesGrid.className = "grid grid-cols-2 gap-2 mb-4";

    // Create reference frames
    for (let i = 1; i <= 4; i++) {
      const frameContainer = document.createElement("div");
      frameContainer.className =
        "bg-gray-200 rounded-lg p-1 flex items-center justify-center";
      frameContainer.style.width = "80px";
      frameContainer.style.height = "80px";

      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("width", "70");
      svg.setAttribute("height", "70");
      svg.setAttribute("viewBox", "0 0 70 70");

      // Circles
      const circle1 = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "circle"
      );
      circle1.setAttribute("cx", "35");
      circle1.setAttribute("cy", "35");
      circle1.setAttribute("r", 25 - i * 2);
      circle1.setAttribute("fill", "rgba(30, 64, 175, 0.15)");
      svg.appendChild(circle1);

      const circle2 = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "circle"
      );
      circle2.setAttribute("cx", "35");
      circle2.setAttribute("cy", "35");
      circle2.setAttribute("r", 15 - i);
      circle2.setAttribute("fill", "rgba(30, 64, 175, 0.25)");
      svg.appendChild(circle2);

      const circle3 = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "circle"
      );
      circle3.setAttribute("cx", "35");
      circle3.setAttribute("cy", "35");
      circle3.setAttribute("r", 8 - i / 4);
      circle3.setAttribute("fill", "rgba(30, 64, 175, 0.35)");
      svg.appendChild(circle3);

      // Speckles
      for (let j = 0; j < 8; j++) {
        const angle = Math.random() * Math.PI * 2;
        const radius = 7 + Math.random() * 20;
        const x = 35 + Math.cos(angle) * radius;
        const y = 35 + Math.sin(angle) * radius;

        const speckle = document.createElementNS(
          "http://www.w3.org/2000/svg",
          "circle"
        );
        speckle.setAttribute("cx", x);
        speckle.setAttribute("cy", y);
        speckle.setAttribute("r", 0.5 + Math.random() * 1.5);
        speckle.setAttribute(
          "fill",
          `rgba(30, 64, 175, ${0.3 + Math.random() * 0.4})`
        );
        svg.appendChild(speckle);
      }

      frameContainer.appendChild(svg);
      framesGrid.appendChild(frameContainer);
    }

    framesContainer.appendChild(framesGrid);

    const framesLabel = document.createElement("div");
    framesLabel.className = "text-center text-sm font-medium mb-2";
    framesLabel.textContent = "Processed Reference Frames";
    framesContainer.appendChild(framesLabel);

    vizContainer.appendChild(framesContainer);

    // Arrow
    const arrowContainer = document.createElement("div");
    arrowContainer.className = "flex items-center";

    const arrowSvg = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "svg"
    );
    arrowSvg.setAttribute("width", "28");
    arrowSvg.setAttribute("height", "28");
    arrowSvg.setAttribute("viewBox", "0 0 24 24");
    arrowSvg.setAttribute("fill", "none");
    arrowSvg.setAttribute("stroke", "currentColor");
    arrowSvg.setAttribute("stroke-width", "2");
    arrowSvg.setAttribute("stroke-linecap", "round");
    arrowSvg.setAttribute("stroke-linejoin", "round");
    arrowSvg.className = "text-blue-600";

    const arrowPath = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "path"
    );
    arrowPath.setAttribute("d", "M5 12h14");
    arrowSvg.appendChild(arrowPath);

    const arrowHead = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "path"
    );
    arrowHead.setAttribute("d", "M12 5l7 7-7 7");
    arrowSvg.appendChild(arrowHead);

    arrowContainer.appendChild(arrowSvg);
    vizContainer.appendChild(arrowContainer);

    // Matrix
    const matrixContainer = document.createElement("div");
    matrixContainer.className = "flex flex-col items-center";

    const matrix = document.createElement("div");
    matrix.className =
      "bg-blue-100 border-2 border-blue-600 p-2 rounded-lg w-64 h-48 flex items-center justify-center";

    const matrixGrid = document.createElement("div");
    matrixGrid.className = "grid grid-cols-6 gap-px";

    // Create matrix cells
    for (let i = 0; i < 60; i++) {
      const cell = document.createElement("div");
      cell.className = "w-4 h-4";
      cell.style.backgroundColor = `rgba(30, 64, 175, ${
        0.1 + Math.random() * 0.3
      })`;
      matrixGrid.appendChild(cell);
    }

    matrix.appendChild(matrixGrid);
    matrixContainer.appendChild(matrix);

    const matrixLabel = document.createElement("div");
    matrixLabel.className = "text-center text-sm font-medium mt-2";
    matrixLabel.textContent = "Reference PSF Matrix";
    matrixContainer.appendChild(matrixLabel);

    vizContainer.appendChild(matrixContainer);
    stepDiv.appendChild(vizContainer);

    // Description
    const description = document.createElement("p");
    description.className = "text-gray-700 text-center max-w-2xl";
    description.textContent =
      "Each reference frame is processed (mean-subtraction, normalization) and then flattened into a 1D vector. These vectors are combined to form a matrix where each column represents a reference PSF. This matrix captures the PSF variability across the reference library.";
    stepDiv.appendChild(description);

    contentContainer.appendChild(stepDiv);
  }

  function renderStep3() {
    const stepDiv = document.createElement("div");
    stepDiv.className = "flex flex-col items-center";

    const heading = document.createElement("h3");
    heading.className = "text-xl font-semibold mb-4";
    heading.textContent = "Singular Value Decomposition (SVD)";
    stepDiv.appendChild(heading);

    // SVD visualization
    const svdContainer = document.createElement("div");
    svdContainer.className = "flex items-center justify-center space-x-4 mb-6";

    // Matrix A
    const matrixAContainer = document.createElement("div");
    matrixAContainer.className = "flex flex-col items-center";

    const matrixA = document.createElement("div");
    matrixA.className =
      "bg-blue-100 border-2 border-blue-600 p-2 rounded-lg w-48 h-64 flex items-center justify-center";

    const matrixAGrid = document.createElement("div");
    matrixAGrid.className = "grid grid-cols-5 gap-px";

    for (let i = 0; i < 50; i++) {
      const cell = document.createElement("div");
      cell.className = "w-4 h-4";
      cell.style.backgroundColor = `rgba(30, 64, 175, ${
        0.1 + Math.random() * 0.3
      })`;
      matrixAGrid.appendChild(cell);
    }

    matrixA.appendChild(matrixAGrid);
    matrixAContainer.appendChild(matrixA);

    const matrixALabel = document.createElement("div");
    matrixALabel.className = "text-center text-sm font-medium mt-2";
    matrixALabel.textContent = "Matrix A";
    matrixAContainer.appendChild(matrixALabel);

    svdContainer.appendChild(matrixAContainer);

    // Equals sign
    const equalsSign = document.createElement("div");
    equalsSign.className = "text-3xl font-bold text-blue-700";
    equalsSign.textContent = "=";
    svdContainer.appendChild(equalsSign);

    // Matrix U
    const matrixUContainer = document.createElement("div");
    matrixUContainer.className = "flex flex-col items-center";

    const matrixU = document.createElement("div");
    matrixU.className =
      "bg-red-100 border-2 border-red-600 p-2 rounded-lg w-48 h-64 flex items-center justify-center";

    const matrixUGrid = document.createElement("div");
    matrixUGrid.className = "grid grid-cols-5 gap-px";

    for (let i = 0; i < 50; i++) {
      const cell = document.createElement("div");
      cell.className = "w-4 h-4";
      cell.style.backgroundColor = `rgba(220, 38, 38, ${
        0.1 + Math.random() * 0.3
      })`;
      matrixUGrid.appendChild(cell);
    }

    matrixU.appendChild(matrixUGrid);
    matrixUContainer.appendChild(matrixU);

    const matrixULabel = document.createElement("div");
    matrixULabel.className = "text-center text-sm font-medium mt-2";
    matrixULabel.textContent = "Matrix U";
    matrixUContainer.appendChild(matrixULabel);

    svdContainer.appendChild(matrixUContainer);

    // Matrix Sigma
    const matrixSigmaContainer = document.createElement("div");
    matrixSigmaContainer.className = "flex flex-col items-center";

    const matrixSigma = document.createElement("div");
    matrixSigma.className =
      "bg-green-100 border-2 border-green-600 p-2 rounded-lg w-32 h-32 flex items-center justify-center";

    const sigmaContent = document.createElement("div");
    sigmaContent.className = "relative w-full h-full";

    // Diagonal line
    const diagonalLine = document.createElement("div");
    diagonalLine.className =
      "absolute top-0 left-0 w-full h-full flex justify-center items-center";

    const line = document.createElement("div");
    line.className = "w-28 h-4 bg-green-600 transform rotate-45 rounded-full";
    diagonalLine.appendChild(line);
    sigmaContent.appendChild(diagonalLine);

    // Singular values
    for (let i = 0; i < 5; i++) {
      const singularValue = document.createElement("div");
      singularValue.className = "absolute w-4 h-4 bg-green-800 rounded-full";
      singularValue.style.top = `${16 + i * 16}px`;
      singularValue.style.left = `${16 + i * 16}px`;
      sigmaContent.appendChild(singularValue);
    }

    matrixSigma.appendChild(sigmaContent);
    matrixSigmaContainer.appendChild(matrixSigma);

    const matrixSigmaLabel = document.createElement("div");
    matrixSigmaLabel.className = "text-center text-sm font-medium mt-2";
    matrixSigmaLabel.innerHTML = "Matrix Σ";
    matrixSigmaContainer.appendChild(matrixSigmaLabel);

    svdContainer.appendChild(matrixSigmaContainer);

    // Matrix V^T
    const matrixVTContainer = document.createElement("div");
    matrixVTContainer.className = "flex flex-col items-center";

    const matrixVT = document.createElement("div");
    matrixVT.className =
      "bg-yellow-100 border-2 border-yellow-600 p-2 rounded-lg w-32 h-48 flex items-center justify-center";

    const matrixVTGrid = document.createElement("div");
    matrixVTGrid.className = "grid grid-cols-3 gap-px";

    for (let i = 0; i < 30; i++) {
      const cell = document.createElement("div");
      cell.className = "w-4 h-4";
      cell.style.backgroundColor = `rgba(202, 138, 4, ${
        0.1 + Math.random() * 0.3
      })`;
      matrixVTGrid.appendChild(cell);
    }

    matrixVT.appendChild(matrixVTGrid);
    matrixVTContainer.appendChild(matrixVT);

    const matrixVTLabel = document.createElement("div");
    matrixVTLabel.className = "text-center text-sm font-medium mt-2";
    matrixVTLabel.innerHTML = "Matrix V<sup>T</sup>";
    matrixVTContainer.appendChild(matrixVTLabel);

    svdContainer.appendChild(matrixVTContainer);
    stepDiv.appendChild(svdContainer);

    // Description
    const description = document.createElement("p");
    description.className = "text-gray-700 text-center max-w-2xl";
    description.innerHTML =
      "Singular Value Decomposition (SVD) decomposes the reference PSF matrix into three components: U (left singular vectors), Σ (singular values), and V<sup>T</sup> (right singular vectors). The columns of U represent the KL basis vectors (eigenimages) that efficiently capture the PSF variability.";
    stepDiv.appendChild(description);

    contentContainer.appendChild(stepDiv);
  }

  function renderStep4() {
    const stepDiv = document.createElement("div");
    stepDiv.className = "flex flex-col items-center";

    const heading = document.createElement("h3");
    heading.className = "text-xl font-semibold mb-4";
    heading.textContent = "KL Basis Vectors (Eigenimages)";
    stepDiv.appendChild(heading);

    // KL modes grid
    const klModesGrid = document.createElement("div");
    klModesGrid.className = "grid grid-cols-3 gap-6 mb-6";

    // Create KL modes
    for (let i = 1; i <= 6; i++) {
      const modeContainer = document.createElement("div");
      modeContainer.className = "flex flex-col items-center";

      const modeVisual = document.createElement("div");
      modeVisual.className =
        "bg-gray-200 rounded-lg p-2 flex items-center justify-center";
      modeVisual.style.width = "120px";
      modeVisual.style.height = "120px";

      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("width", "100");
      svg.setAttribute("height", "100");
      svg.setAttribute("viewBox", "0 0 100 100");

      // Different patterns for each mode
      if (i === 1) {
        // Concentric circles
        for (let r = 40; r >= 10; r -= 10) {
          const circle = document.createElementNS(
            "http://www.w3.org/2000/svg",
            "circle"
          );
          circle.setAttribute("cx", "50");
          circle.setAttribute("cy", "50");
          circle.setAttribute("r", r);
          circle.setAttribute(
            "fill",
            `rgba(30, 64, 175, ${0.1 + (40 - r) * 0.005})`
          );
          svg.appendChild(circle);
        }
      } else if (i === 2) {
        // Two circles
        const circle1 = document.createElementNS(
          "http://www.w3.org/2000/svg",
          "circle"
        );
        circle1.setAttribute("cx", "40");
        circle1.setAttribute("cy", "40");
        circle1.setAttribute("r", "20");
        circle1.setAttribute("fill", "rgba(30, 64, 175, 0.2)");
        svg.appendChild(circle1);

        const circle2 = document.createElementNS(
          "http://www.w3.org/2000/svg",
          "circle"
        );
        circle2.setAttribute("cx", "60");
        circle2.setAttribute("cy", "60");
        circle2.setAttribute("r", "20");
        circle2.setAttribute("fill", "rgba(30, 64, 175, 0.2)");
        svg.appendChild(circle2);
      } else if (i === 3) {
        // Horizontal pair
        const circle1 = document.createElementNS(
          "http://www.w3.org/2000/svg",
          "circle"
        );
        circle1.setAttribute("cx", "30");
        circle1.setAttribute("cy", "50");
        circle1.setAttribute("r", "15");
        circle1.setAttribute("fill", "rgba(30, 64, 175, 0.2)");
        svg.appendChild(circle1);

        const circle2 = document.createElementNS(
          "http://www.w3.org/2000/svg",
          "circle"
        );
        circle2.setAttribute("cx", "70");
        circle2.setAttribute("cy", "50");
        circle2.setAttribute("r", "15");
        circle2.setAttribute("fill", "rgba(30, 64, 175, 0.2)");
        svg.appendChild(circle2);
      } else if (i === 4) {
        // Vertical pair
        const circle1 = document.createElementNS(
          "http://www.w3.org/2000/svg",
          "circle"
        );
        circle1.setAttribute("cx", "50");
        circle1.setAttribute("cy", "30");
        circle1.setAttribute("r", "15");
        circle1.setAttribute("fill", "rgba(30, 64, 175, 0.2)");
        svg.appendChild(circle1);

        const circle2 = document.createElementNS(
          "http://www.w3.org/2000/svg",
          "circle"
        );
        circle2.setAttribute("cx", "50");
        circle2.setAttribute("cy", "70");
        circle2.setAttribute("r", "15");
        circle2.setAttribute("fill", "rgba(30, 64, 175, 0.2)");
        svg.appendChild(circle2);
      } else if (i === 5) {
        // Four corners
        for (let x = 30; x <= 70; x += 40) {
          for (let y = 30; y <= 70; y += 40) {
            const circle = document.createElementNS(
              "http://www.w3.org/2000/svg",
              "circle"
            );
            circle.setAttribute("cx", x);
            circle.setAttribute("cy", y);
            circle.setAttribute("r", "12");
            circle.setAttribute("fill", "rgba(30, 64, 175, 0.2)");
            svg.appendChild(circle);
          }
        }
      } else if (i === 6) {
        // Radial pattern
        const bgCircle = document.createElementNS(
          "http://www.w3.org/2000/svg",
          "circle"
        );
        bgCircle.setAttribute("cx", "50");
        bgCircle.setAttribute("cy", "50");
        bgCircle.setAttribute("r", "40");
        bgCircle.setAttribute("fill", "rgba(30, 64, 175, 0.05)");
        svg.appendChild(bgCircle);

        for (let j = 0; j < 12; j++) {
          const angle = (j * Math.PI * 2) / 12;
          const x = 50 + Math.cos(angle) * 30;
          const y = 50 + Math.sin(angle) * 30;

          const spot = document.createElementNS(
            "http://www.w3.org/2000/svg",
            "circle"
          );
          spot.setAttribute("cx", x);
          spot.setAttribute("cy", y);
          spot.setAttribute("r", "6");
          spot.setAttribute(
            "fill",
            `rgba(30, 64, 175, ${0.15 + (j % 2) * 0.1})`
          );
          svg.appendChild(spot);
        }
      }

      modeVisual.appendChild(svg);
      modeContainer.appendChild(modeVisual);

      const modeLabel = document.createElement("div");
      modeLabel.className = "text-center text-sm font-medium mt-2";
      modeLabel.textContent = `KL Mode ${i}`;
      modeContainer.appendChild(modeLabel);

      klModesGrid.appendChild(modeContainer);
    }

    stepDiv.appendChild(klModesGrid);

    // Description
    const description = document.createElement("p");
    description.className = "text-gray-700 text-center max-w-2xl";
    description.textContent =
      "The resulting KL basis vectors (eigenimages) represent different modes of PSF variation. The first few modes capture the most significant variations in the PSF structure. These basis vectors are ordered by their importance (singular values) in representing the PSF variability across the reference library.";
    stepDiv.appendChild(description);

    // Code snippet
    const codeBox = document.createElement("div");
    codeBox.className = "mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg";

    const codeContent = document.createElement("div");
    codeContent.className = "text-sm text-blue-800";

    const codeSnippet = document.createElement("code");
    codeSnippet.className = "bg-blue-100 px-2 py-1 rounded font-mono";
    codeSnippet.textContent = "Z_KL = basis_fn(ref_flat, K_max)";
    codeContent.appendChild(codeSnippet);

    const lineBreak = document.createElement("br");
    codeContent.appendChild(lineBreak);
    codeContent.appendChild(document.createTextNode("Where:"));

    const paramsList = document.createElement("ul");
    paramsList.className = "list-disc list-inside mt-2";

    const params = [
      { param: "Z_KL", desc: "The computed KL basis vectors" },
      { param: "ref_flat", desc: "Flattened reference PSF library" },
      { param: "K_max", desc: "Maximum number of KL modes to compute" },
    ];

    params.forEach((item) => {
      const listItem = document.createElement("li");
      const paramCode = document.createElement("code");
      paramCode.className = "bg-blue-100 px-1 rounded font-mono";
      paramCode.textContent = item.param;

      listItem.appendChild(paramCode);
      listItem.appendChild(document.createTextNode(`: ${item.desc}`));
      paramsList.appendChild(listItem);
    });

    codeContent.appendChild(paramsList);
    codeBox.appendChild(codeContent);
    stepDiv.appendChild(codeBox);

    contentContainer.appendChild(stepDiv);
  }

  // Initialize the UI
  updateUI();
});

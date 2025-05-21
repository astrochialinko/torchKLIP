document.addEventListener("DOMContentLoaded", function () {
  // Target the div with id="step1"
  const targetDiv = document.getElementById("step1");

  if (!targetDiv) {
    console.error("Target div with id 'step1' not found!");
    return;
  }

  // Initialize step state
  let currentStep = 1;
  const totalSteps = 4;

  // Create the main container
  const mainContainer = document.createElement("div");
  mainContainer.className = "klip-visualization";
  mainContainer.style.fontFamily = "Arial, sans-serif";
  mainContainer.style.maxWidth = "800px";
  mainContainer.style.margin = "0 auto";
  mainContainer.style.padding = "20px";

  // Create visualization container
  const vizContainer = document.createElement("div");
  vizContainer.className = "visualization-container";
  vizContainer.style.position = "relative";
  vizContainer.style.height = "400px";
  vizContainer.style.backgroundColor = "#f9f9f9";
  vizContainer.style.border = "1px solid #e0e0e0";
  vizContainer.style.borderRadius = "8px";
  vizContainer.style.padding = "20px";
  vizContainer.style.marginBottom = "20px";
  vizContainer.style.overflow = "hidden";

  // Create navigation controls
  const navControls = document.createElement("div");
  navControls.className = "nav-controls";
  navControls.style.display = "flex";
  navControls.style.justifyContent = "space-between";
  navControls.style.alignItems = "center";
  navControls.style.marginBottom = "20px";

  // Previous button
  const prevButton = document.createElement("button");
  prevButton.textContent = "Previous Step";
  prevButton.disabled = true;
  prevButton.style.padding = "8px 16px";
  prevButton.style.backgroundColor = "#dddddd";
  prevButton.style.border = "none";
  prevButton.style.borderRadius = "4px";
  prevButton.style.cursor = "not-allowed";

  // Step indicator
  const stepIndicator = document.createElement("div");
  stepIndicator.textContent = `Step ${currentStep} of ${totalSteps}`;
  stepIndicator.style.fontWeight = "bold";

  // Next button
  const nextButton = document.createElement("button");
  nextButton.textContent = "Next Step";
  nextButton.style.padding = "8px 16px";
  nextButton.style.backgroundColor = "#4a86e8";
  nextButton.style.color = "white";
  nextButton.style.border = "none";
  nextButton.style.borderRadius = "4px";
  nextButton.style.cursor = "pointer";

  // UI structure
  navControls.appendChild(prevButton);
  navControls.appendChild(stepIndicator);
  navControls.appendChild(nextButton);

  mainContainer.appendChild(vizContainer);
  mainContainer.appendChild(navControls);

  targetDiv.appendChild(mainContainer);

  // Function to create 3D datacube visualization
  function createDatacubeViz() {
    const container = document.createElement("div");
    container.style.display = "flex";
    container.style.flexDirection = "column";
    container.style.alignItems = "center";
    container.style.height = "100%";

    const stepTitle = document.createElement("h3");
    stepTitle.textContent = "Step 1: 3D Datacube";
    stepTitle.style.marginBottom = "20px";
    stepTitle.style.fontSize = "20px";

    const cubeContainer = document.createElement("div");
    cubeContainer.style.position = "relative";
    cubeContainer.style.height = "240px";
    cubeContainer.style.width = "240px";

    // Create the cube layers
    for (let layer = 0; layer < 3; layer++) {
      const gridLayer = document.createElement("div");
      gridLayer.style.position = "absolute";
      gridLayer.style.left = `${layer * 20}px`;
      gridLayer.style.top = `${layer * 20}px`;
      gridLayer.style.display = "grid";
      gridLayer.style.gridTemplateColumns = "repeat(4, 1fr)";
      gridLayer.style.gap = "4px";

      // Create cells for this layer
      for (let i = 0; i < 16; i++) {
        const cell = document.createElement("div");
        cell.style.width = "40px";
        cell.style.height = "40px";
        cell.style.backgroundColor = `rgba(66, 133, 244, ${0.6 + layer * 0.2})`;
        cell.style.borderRadius = "4px";
        cell.style.display = "flex";
        cell.style.alignItems = "center";
        cell.style.justifyContent = "center";
        cell.style.fontSize = "12px";
        cell.textContent = `Frame ${layer + 1}`;

        gridLayer.appendChild(cell);
      }

      cubeContainer.appendChild(gridLayer);
    }

    const description = document.createElement("p");
    description.textContent = "Original 3D Datacube: (frames × height × width)";
    description.style.marginTop = "30px";
    description.style.textAlign = "center";
    description.style.color = "#555";

    container.appendChild(stepTitle);
    container.appendChild(cubeContainer);
    container.appendChild(description);

    return container;
  }

  // Function to create mean subtraction visualization
  function createMeanSubtractionViz() {
    const container = document.createElement("div");
    container.style.display = "flex";
    container.style.flexDirection = "column";
    container.style.alignItems = "center";
    container.style.height = "100%";

    const stepTitle = document.createElement("h3");
    stepTitle.textContent = "Step 2: Mean Subtraction";
    stepTitle.style.marginBottom = "20px";
    stepTitle.style.fontSize = "20px";

    const rowContainer = document.createElement("div");
    rowContainer.style.display = "flex";
    rowContainer.style.alignItems = "center";
    rowContainer.style.gap = "30px";

    // Create original data representation
    const originalData = document.createElement("div");
    originalData.style.position = "relative";
    originalData.style.height = "180px";
    originalData.style.width = "180px";

    // Create the original data layers
    for (let layer = 0; layer < 2; layer++) {
      const gridLayer = document.createElement("div");
      gridLayer.style.position = "absolute";
      gridLayer.style.left = `${layer * 15}px`;
      gridLayer.style.top = `${layer * 15}px`;
      gridLayer.style.display = "grid";
      gridLayer.style.gridTemplateColumns = "repeat(4, 1fr)";
      gridLayer.style.gap = "3px";

      // Create cells with random values
      for (let i = 0; i < 16; i++) {
        const cell = document.createElement("div");
        cell.style.width = "35px";
        cell.style.height = "35px";
        cell.style.backgroundColor = `rgba(66, 133, 244, ${0.7 + layer * 0.2})`;
        cell.style.borderRadius = "4px";
        cell.style.display = "flex";
        cell.style.alignItems = "center";
        cell.style.justifyContent = "center";
        cell.style.fontSize = "11px";

        // Generate a random value between 0 and 10
        const value = (Math.random() * 10).toFixed(1);
        cell.textContent = value;

        gridLayer.appendChild(cell);
      }

      originalData.appendChild(gridLayer);
    }

    // Create arrow symbol
    const arrow = document.createElement("div");
    arrow.textContent = "→";
    arrow.style.fontSize = "32px";
    arrow.style.fontWeight = "bold";

    // Create mean-subtracted data
    const subtractedData = document.createElement("div");
    subtractedData.style.position = "relative";
    subtractedData.style.height = "180px";
    subtractedData.style.width = "180px";

    // Create the mean-subtracted data layers
    for (let layer = 0; layer < 2; layer++) {
      const gridLayer = document.createElement("div");
      gridLayer.style.position = "absolute";
      gridLayer.style.left = `${layer * 15}px`;
      gridLayer.style.top = `${layer * 15}px`;
      gridLayer.style.display = "grid";
      gridLayer.style.gridTemplateColumns = "repeat(4, 1fr)";
      gridLayer.style.gap = "3px";

      // Create cells with random values centered around 0
      for (let i = 0; i < 16; i++) {
        const cell = document.createElement("div");
        cell.style.width = "35px";
        cell.style.height = "35px";
        cell.style.backgroundColor = `rgba(76, 175, 80, ${0.7 + layer * 0.2})`;
        cell.style.borderRadius = "4px";
        cell.style.display = "flex";
        cell.style.alignItems = "center";
        cell.style.justifyContent = "center";
        cell.style.fontSize = "11px";

        // Generate a random value between -1 and 1
        const value = (Math.random() * 2 - 1).toFixed(1);
        cell.textContent = value;

        gridLayer.appendChild(cell);
      }

      subtractedData.appendChild(gridLayer);
    }

    rowContainer.appendChild(originalData);
    rowContainer.appendChild(arrow);
    rowContainer.appendChild(subtractedData);

    const description = document.createElement("p");
    description.textContent =
      "Mean value subtracted from each pixel location across all frames";
    description.style.marginTop = "10px";
    description.style.textAlign = "center";
    description.style.color = "#555";

    container.appendChild(stepTitle);
    container.appendChild(rowContainer);
    container.appendChild(description);

    return container;
  }

  // Function to create NaN replacement visualization
  function createNanReplacementViz() {
    const container = document.createElement("div");
    container.style.display = "flex";
    container.style.flexDirection = "column";
    container.style.alignItems = "center";
    container.style.height = "100%";

    const stepTitle = document.createElement("h3");
    stepTitle.textContent = "Step 3: NaN Replacement";
    stepTitle.style.marginBottom = "20px";
    stepTitle.style.fontSize = "20px";

    const rowContainer = document.createElement("div");
    rowContainer.style.display = "flex";
    rowContainer.style.alignItems = "center";
    rowContainer.style.gap = "30px";

    // Create data with NaNs
    const dataWithNans = document.createElement("div");
    dataWithNans.style.display = "grid";
    dataWithNans.style.gridTemplateColumns = "repeat(4, 1fr)";
    dataWithNans.style.gap = "4px";

    // Create cells with some NaN values
    for (let i = 0; i < 16; i++) {
      const cell = document.createElement("div");
      cell.style.width = "40px";
      cell.style.height = "40px";

      const isNaN = i % 5 === 0;
      cell.style.backgroundColor = isNaN ? "#ffcdd2" : "#a5d6a7";
      cell.style.borderRadius = "4px";
      cell.style.display = "flex";
      cell.style.alignItems = "center";
      cell.style.justifyContent = "center";
      cell.style.fontSize = "12px";

      if (isNaN) {
        cell.textContent = "NaN";
      } else {
        // Generate a random value between -1 and 1
        cell.textContent = (Math.random() * 2 - 1).toFixed(1);
      }

      dataWithNans.appendChild(cell);
    }

    // Create arrow symbol
    const arrow = document.createElement("div");
    arrow.textContent = "→";
    arrow.style.fontSize = "32px";
    arrow.style.fontWeight = "bold";

    // Create NaN-replaced data
    const replacedData = document.createElement("div");
    replacedData.style.display = "grid";
    replacedData.style.gridTemplateColumns = "repeat(4, 1fr)";
    replacedData.style.gap = "4px";

    // Create cells with NaNs replaced by 0.0
    for (let i = 0; i < 16; i++) {
      const cell = document.createElement("div");
      cell.style.width = "40px";
      cell.style.height = "40px";

      const wasNaN = i % 5 === 0;
      cell.style.backgroundColor = wasNaN ? "#fff9c4" : "#a5d6a7";
      cell.style.borderRadius = "4px";
      cell.style.display = "flex";
      cell.style.alignItems = "center";
      cell.style.justifyContent = "center";
      cell.style.fontSize = "12px";

      if (wasNaN) {
        cell.textContent = "0.0";
      } else {
        // Generate a random value between -1 and 1
        cell.textContent = (Math.random() * 2 - 1).toFixed(1);
      }

      replacedData.appendChild(cell);
    }

    rowContainer.appendChild(dataWithNans);
    rowContainer.appendChild(arrow);
    rowContainer.appendChild(replacedData);

    const description = document.createElement("p");
    description.textContent = "All NaN values are replaced with zeros";
    description.style.marginTop = "10px";
    description.style.textAlign = "center";
    description.style.color = "#555";

    container.appendChild(stepTitle);
    container.appendChild(rowContainer);
    container.appendChild(description);

    return container;
  }

  // Function to create flattening visualization
  function createFlatteningViz() {
    const container = document.createElement("div");
    container.style.display = "flex";
    container.style.flexDirection = "column";
    container.style.alignItems = "center";
    container.style.height = "100%";

    const stepTitle = document.createElement("h3");
    stepTitle.textContent = "Step 4: Flattening";
    stepTitle.style.marginBottom = "20px";
    stepTitle.style.fontSize = "20px";

    const rowContainer = document.createElement("div");
    rowContainer.style.display = "flex";
    rowContainer.style.alignItems = "center";
    rowContainer.style.justifyContent = "center"; // Add center alignment
    rowContainer.style.gap = "30px";

    // Create 3D data
    const threeDData = document.createElement("div");
    threeDData.style.display = "flex";
    threeDData.style.flexDirection = "column";
    threeDData.style.gap = "8px";

    // Create 3 frames
    const colors = ["#a5d6a7", "#81c784", "#66bb6a"];
    for (let frame = 0; frame < 3; frame++) {
      const frameGrid = document.createElement("div");
      frameGrid.style.display = "grid";
      frameGrid.style.gridTemplateColumns = "repeat(4, 1fr)";
      frameGrid.style.gap = "3px";

      // Create cells for this frame
      for (let i = 0; i < 16; i++) {
        const cell = document.createElement("div");
        cell.style.width = "30px";
        cell.style.height = "30px";
        cell.style.backgroundColor = colors[frame];
        cell.style.borderRadius = "4px";
        cell.style.display = "flex";
        cell.style.alignItems = "center";
        cell.style.justifyContent = "center";
        cell.style.fontSize = "10px";
        cell.textContent = `Fr${frame + 1}`;

        frameGrid.appendChild(cell);
      }

      threeDData.appendChild(frameGrid);
    }

    // Create arrow symbol
    const arrow = document.createElement("div");
    arrow.textContent = "→";
    arrow.style.fontSize = "32px";
    arrow.style.fontWeight = "bold";

    // Create 2D matrix visualization
    const flattenedData = document.createElement("div");
    flattenedData.style.border = "1px solid #ddd";
    flattenedData.style.borderRadius = "4px";
    flattenedData.style.overflow = "hidden";
    flattenedData.style.width = "300px";
    flattenedData.style.alignSelf = "center"; // Add this for vertical alignment
    flattenedData.style.marginTop = "-90px"; // Move up by 20px

    // Create rows for the frames
    for (let frame = 0; frame < 3; frame++) {
      const row = document.createElement("div");
      row.style.display = "flex";
      row.style.alignItems = "center";
      row.style.backgroundColor = "#e1bee7";
      row.style.padding = "6px";
      row.style.marginBottom = "2px";

      // Frame label
      const label = document.createElement("div");
      label.style.width = "80px";
      label.style.textAlign = "center";
      label.style.fontWeight = "bold";
      label.style.fontSize = "14px";
      label.textContent = `Frame ${frame + 1}`;

      // Pixel array representation
      const pixelArray = document.createElement("div");
      pixelArray.style.flex = "1";
      pixelArray.style.backgroundColor = "#ce93d8";
      pixelArray.style.height = "32px";
      pixelArray.style.display = "flex";
      pixelArray.style.alignItems = "center";
      pixelArray.style.padding = "0 10px";
      pixelArray.style.borderRadius = "4px";
      pixelArray.style.fontSize = "14px";
      pixelArray.textContent = `[p${frame + 1}₁, p${frame + 1}₂, p${
        frame + 1
      }₃, ..., p${frame + 1}ₙ]`;

      row.appendChild(label);
      row.appendChild(pixelArray);
      flattenedData.appendChild(row);
    }

    // Add ellipsis row
    const ellipsisRow = document.createElement("div");
    ellipsisRow.style.backgroundColor = "#f3e5f5";
    ellipsisRow.style.height = "30px";
    ellipsisRow.style.display = "flex";
    ellipsisRow.style.alignItems = "center";
    ellipsisRow.style.justifyContent = "center";
    ellipsisRow.style.color = "#777";
    ellipsisRow.textContent = "⋮";
    flattenedData.appendChild(ellipsisRow);

    // Add last row
    const lastRow = document.createElement("div");
    lastRow.style.display = "flex";
    lastRow.style.alignItems = "center";
    lastRow.style.backgroundColor = "#e1bee7";
    lastRow.style.padding = "6px";

    const lastLabel = document.createElement("div");
    lastLabel.style.width = "80px";
    lastLabel.style.textAlign = "center";
    lastLabel.style.fontWeight = "bold";
    lastLabel.style.fontSize = "14px";
    lastLabel.textContent = "Frame k";

    const lastPixelArray = document.createElement("div");
    lastPixelArray.style.flex = "1";
    lastPixelArray.style.backgroundColor = "#ce93d8";
    lastPixelArray.style.height = "32px";
    lastPixelArray.style.display = "flex";
    lastPixelArray.style.alignItems = "center";
    lastPixelArray.style.padding = "0 10px";
    lastPixelArray.style.borderRadius = "4px";
    lastPixelArray.style.fontSize = "14px";
    lastPixelArray.textContent = "[pₖ₁, pₖ₂, pₖ₃, ..., pₖₙ]";

    lastRow.appendChild(lastLabel);
    lastRow.appendChild(lastPixelArray);
    flattenedData.appendChild(lastRow);

    rowContainer.appendChild(threeDData);
    rowContainer.appendChild(arrow);
    rowContainer.appendChild(flattenedData);

    const description = document.createElement("p");
    description.textContent =
      "3D datacube flattened to 2D matrix: (frames × [height·width])";
    description.style.marginTop = "10px";
    description.style.textAlign = "center";
    description.style.color = "#555";

    container.appendChild(stepTitle);
    container.appendChild(rowContainer);
    container.appendChild(description);

    return container;
  }

  // Function to update the visualization based on the current step
  function updateVisualization() {
    // Clear previous visualization
    vizContainer.innerHTML = "";

    // Add new visualization based on step
    let visualization;
    switch (currentStep) {
      case 1:
        visualization = createDatacubeViz();
        break;
      case 2:
        visualization = createMeanSubtractionViz();
        break;
      case 3:
        visualization = createNanReplacementViz();
        break;
      case 4:
        visualization = createFlatteningViz();
        break;
      default:
        visualization = createDatacubeViz();
    }

    vizContainer.appendChild(visualization);

    // Update step indicator
    stepIndicator.textContent = `Step ${currentStep} of ${totalSteps}`;

    // Update button states
    prevButton.disabled = currentStep === 1;
    nextButton.disabled = currentStep === totalSteps;

    if (prevButton.disabled) {
      prevButton.style.backgroundColor = "#dddddd";
      prevButton.style.cursor = "not-allowed";
    } else {
      prevButton.style.backgroundColor = "#4a86e8";
      prevButton.style.color = "white";
      prevButton.style.cursor = "pointer";
    }

    if (nextButton.disabled) {
      nextButton.style.backgroundColor = "#dddddd";
      nextButton.style.cursor = "not-allowed";
    } else {
      nextButton.style.backgroundColor = "#4a86e8";
      nextButton.style.color = "white";
      nextButton.style.cursor = "pointer";
    }
  }

  // Add event listeners to buttons
  prevButton.addEventListener("click", function () {
    if (currentStep > 1) {
      currentStep--;
      updateVisualization();
    }
  });

  nextButton.addEventListener("click", function () {
    if (currentStep < totalSteps) {
      currentStep++;
      updateVisualization();
    }
  });

  // Initialize the visualization
  updateVisualization();

  // Add some basic styles
  const styleElement = document.createElement("style");
  styleElement.textContent = `
        .klip-visualization button:hover:not([disabled]) {
          background-color: #3b78e7 !important;
        }
        
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        
        .visualization-container > div {
          animation: fadeIn 0.5s ease-in-out;
        }
      `;
  document.head.appendChild(styleElement);
});

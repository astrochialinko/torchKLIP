// improved-annular-visualization.js
document.addEventListener("DOMContentLoaded", function () {
  // Set up dimensions
  const width = 800;
  const height = 300;
  const margin = 10;
  const centerX = width / 6;
  const centerY = height / 2;
  const innerRadius = 30;
  const outerRadius = 80;
  const numDivisions = 5; // Number of annular rings to create

  // Detect dark mode in PyData Sphinx Theme
  function isDarkTheme() {
    // Check for PyData Sphinx Theme's data-theme attribute
    const dataTheme = document.documentElement.getAttribute("data-theme");
    if (dataTheme) {
      return dataTheme === "dark";
    }

    // Fallback: Check if body has dark background
    const computedStyle = getComputedStyle(document.body);
    const backgroundColor = computedStyle.backgroundColor;
    const rgb = backgroundColor.match(/\d+/g);
    if (rgb && rgb.length >= 3) {
      // Consider it dark if average RGB value is < 128 (out of 255)
      return (parseInt(rgb[0]) + parseInt(rgb[1]) + parseInt(rgb[2])) / 3 < 128;
    }

    return false;
  }

  // Get color settings based on current theme
  const darkMode = isDarkTheme();
  const textColor = darkMode ? "#ffffff" : "#000000";
  const strokeColor = darkMode ? "#ffffff" : "#000000";
  const secondaryTextColor = darkMode ? "#cccccc" : "#666666";
  const lineColor = darkMode ? "#aaaaaa" : "#999999";

  // Create a MutationObserver to detect theme changes
  const themeObserver = new MutationObserver(function (mutations) {
    for (const mutation of mutations) {
      if (mutation.attributeName === "data-theme") {
        // Theme has changed, refresh the visualization
        d3.select("#annuli").selectAll("*").remove();
        createVisualization();
        break;
      }
    }
  });

  // Start observing theme changes
  themeObserver.observe(document.documentElement, { attributes: true });

  // Function to create the visualization
  function createVisualization() {
    // Update colors based on current theme
    const isDark = isDarkTheme();
    const textColor = isDark ? "#ffffff" : "#000000";
    const strokeColor = isDark ? "#ffffff" : "#000000";
    const secondaryTextColor = isDark ? "#cccccc" : "#666666";
    const lineColor = isDark ? "#aaaaaa" : "#999999";

    // Create SVG
    const svg = d3
      .select("#annuli")
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("style", "max-width: 100%; height: auto;");

    // Add title
    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", 25)
      .attr("text-anchor", "middle")
      .attr("font-size", "16px")
      .attr("font-weight", "bold")
      .attr("fill", textColor)
      .text("Annular Region Splitting Methods");

    // Create three visualization groups
    const uniformGroup = svg
      .append("g")
      .attr("transform", `translate(${centerX}, ${centerY})`);

    const logGroup = svg
      .append("g")
      .attr("transform", `translate(${centerX * 3}, ${centerY})`);

    const linearGroup = svg
      .append("g")
      .attr("transform", `translate(${centerX * 5}, ${centerY})`);

    // Add titles for each method
    uniformGroup
      .append("text")
      .attr("y", -outerRadius - 20)
      .attr("text-anchor", "middle")
      .attr("font-size", "14px")
      .attr("fill", textColor)
      .text("Uniform");

    logGroup
      .append("text")
      .attr("y", -outerRadius - 20)
      .attr("text-anchor", "middle")
      .attr("font-size", "14px")
      .attr("fill", textColor)
      .text("Logarithmic");

    linearGroup
      .append("text")
      .attr("y", -outerRadius - 20)
      .attr("text-anchor", "middle")
      .attr("font-size", "14px")
      .attr("fill", textColor)
      .text("Linear");

    // 1. Uniform divisions
    const uniformRadii = Array.from(
      { length: numDivisions + 1 },
      (_, i) => innerRadius + (outerRadius - innerRadius) * (i / numDivisions)
    );

    // 2. Logarithmic divisions
    const logRadii = Array.from({ length: numDivisions + 1 }, (_, i) => {
      const t = i / numDivisions;
      // Using logarithmic scale for the radius
      const logScale = d3
        .scaleLog()
        .domain([1, Math.E])
        .range([innerRadius, outerRadius]);
      return logScale(1 + t * (Math.E - 1));
    });

    // 3. Linear divisions (increasing width from inner to outer)
    const linearRadii = Array.from({ length: numDivisions + 1 }, (_, i) => {
      const t = i / numDivisions;
      // Quadratic increase in radius
      return innerRadius + (outerRadius - innerRadius) * (t * t);
    });

    // Helper function to draw concentric circles
    const drawConcentricCircles = (group, radii, colorScale) => {
      // Draw outer boundary
      group
        .append("circle")
        .attr("r", outerRadius)
        .attr("fill", "none")
        .attr("stroke", strokeColor)
        .attr("stroke-width", 1);

      // Draw each annular ring from outside in
      for (let i = 0; i < radii.length - 1; i++) {
        const innerR = radii[i];
        const outerR = radii[i + 1];

        // Create annular ring
        const arc = d3
          .arc()
          .innerRadius(innerR)
          .outerRadius(outerR)
          .startAngle(0)
          .endAngle(2 * Math.PI);

        group
          .append("path")
          .attr("d", arc)
          .attr("fill", colorScale(i))
          .attr("stroke", "#fff")
          .attr("stroke-width", 0.5);
      }

      // Add center point
      group.append("circle").attr("r", 2).attr("fill", textColor);
    };

    // Use a color scale appropriate for the current theme
    const colorScale = d3
      .scaleSequential(d3.interpolateBlues)
      .domain([numDivisions - 1, 0]);

    // Draw the three different annular regions
    drawConcentricCircles(uniformGroup, uniformRadii.reverse(), colorScale);
    drawConcentricCircles(logGroup, logRadii.reverse(), colorScale);
    drawConcentricCircles(linearGroup, linearRadii.reverse(), colorScale);

    // Add radius labels to show the differences - IMPROVED VERSION
    const addRadiusLabels = (group, radii, name) => {
      // Draw a horizontal radius line
      group
        .append("line")
        .attr("x1", 0)
        .attr("y1", 0)
        .attr("x2", outerRadius + 20)
        .attr("y2", 0)
        .attr("stroke", lineColor)
        .attr("stroke-dasharray", "3,3");

      // Add radius markers without labels
      radii.forEach((r, i) => {
        // Draw small tick at each radius
        group
          .append("circle")
          .attr("r", r)
          .attr("fill", "none")
          .attr("stroke", secondaryTextColor)
          .attr("stroke-width", 0.5)
          .attr("stroke-dasharray", "2,2");
      });
    };

    // Add labels showing radii values
    addRadiusLabels(uniformGroup, uniformRadii, "Uniform");
    addRadiusLabels(logGroup, logRadii, "Log");
    addRadiusLabels(linearGroup, linearRadii, "Linear");

    // Add explanatory annotations
    svg
      .append("text")
      .attr("x", centerX)
      .attr("y", height - 20)
      .attr("text-anchor", "middle")
      .attr("font-size", "10px")
      .attr("fill", textColor)
      .text("Equal width rings");

    svg
      .append("text")
      .attr("x", centerX * 3)
      .attr("y", height - 20)
      .attr("text-anchor", "middle")
      .attr("font-size", "10px")
      .attr("fill", textColor)
      .text("Logarithmic spacing");

    svg
      .append("text")
      .attr("x", centerX * 5)
      .attr("y", height - 20)
      .attr("text-anchor", "middle")
      .attr("font-size", "10px")
      .attr("fill", textColor)
      .text("Quadratic spacing");
  }

  // Create the initial visualization
  createVisualization();
});

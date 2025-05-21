KLIP Parameter Exploration
==========================

This tutorial explores how different parameter choices affect KLIP results using the Beta Pictoris dataset, with a focus on optimizing Signal-to-Noise Ratio (SNR).

Understanding Key Parameters
----------------------------

KLIP performance depends on several key parameters:

- **K_klip**: Number of KL basis vectors to use for PSF subtraction
- **Inner/Outer Working Angle (IWA/OWA)**: The radial range to process
- **Annuli and sections**: How to divide the image for localized processing




Finding Optimal :math:`K_{\rm klip}` Values
-------------------------------------------

The number of KL modes (K_klip) is perhaps the most important parameter:

.. raw:: html

    <div id="controls" style="width:600px; margin:20px auto;">
        <!-- label on its own line -->
        <div style="text-align:center; font-size:14px; margin-bottom:8px; color:var(--ctrl-text);">
          K value:
        </div>

        <!-- slider + bubble in a relative container -->
        <div style="position:relative; height:40px;">
          <input
            id="klip-slider"
            type="range"
            min="1" max="61" value="1"
            style="--pct:0%;"
          />
          <span
            id="klip-label"
            style="
              position:absolute;
              top:25px;
              left:0;
              background: var(--ctrl-bg);
              color: var(--ctrl-text);
              transform:translateX(-50%);
              padding:2px 6px;
              border-radius:4px;
              font-size:12px;
              pointer-events:none;
            "
          >1</span>
        </div>

        <!-- min / max ticks -->
        <div style="display:flex; justify-content:space-between; font-size:12px; margin-top:8px; color:var(--ctrl-text);">
          <span>1</span><span>61</span>
        </div>

        <style>
            :root[data-theme="light"] {
            --ctrl-bg: #e6e6e6;
            --ctrl-text: #000000;
            --track-filled: #5369FF;
            --track-empty:  #E0E0E0;
            }
            :root[data-theme="dark"] {
            --ctrl-bg: #262626;
            --ctrl-text: #ffffff;
            --track-filled: #88AFFF;
            --track-empty:  #444;
            }

          /* track coloring */
          #controls input[type=range] {
            -webkit-appearance: none;
            width: 100%;
            height: 8px;
            border-radius: 4px;
            background:
                linear-gradient(
                to right,
                var(--track-filled) 0%,
                var(--track-filled) var(--pct),
                var(--track-empty)  var(--pct),
                var(--track-empty) 100%
                );
            outline: none;
            margin: 0;
            position: absolute;
            top: 10px; /* drop the track below the label */
          }
          /* thumb styling */
          #controls input[type=range]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px; height: 16px;
            border-radius: 50%;
            background: #fff;
            border: 2px solid #5369FF;
            margin-top: -4px;
            cursor: pointer;
          }
          #controls input[type=range]:focus { outline: none; }

          #klip-tooltip {
            position: absolute;
            pointer-events: none;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 12px;
            opacity: 0;
            transition: opacity 0.1s;
        }
        </style>
    </div>
    
    <!-- Container for both plots side by side -->
    <div style="display: flex; justify-content: center; max-width: 950px; margin: 0 auto;">
        <!-- Left plot container (KLIP heatmap) -->
        <div style="position: relative;">
            <div id="klip-container" style="position: relative;">
                <div id="klip-tooltip"></div>
            </div>
        </div>
        
        <!-- Right plot container (SNR line plot) -->
        <div id="snr-container" style="position: relative;"></div>
    </div>


Optimizing Annuli and Sections
------------------------------

KLIP performance can be improved by processing the image in annular sections:

.. raw:: html

    <div id="annuli"></div>
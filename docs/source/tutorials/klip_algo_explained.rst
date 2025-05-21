KLIP Algorithm: A Step-by-Step Guide
====================================

This tutorial provides a visual walkthrough of the Karhunen-Loève Image Processing (KLIP) algorithm as implemented in our ``torchklip`` package. 
We will break down the process into its core components, showing intermediate data products to help you understand how KLIP transforms the original data into a final, high-contrast image. 
We will use the well-known Beta Pictoris (``betapic``) exoplanetary system as our example dataset to visualize each stage of the process.



Intuition Behind KLIP
---------------------

.. raw:: html 

   <div id="post_processing" style="text-align: center;">
     <img src="../_static/images/post_processing.png" alt="post processing">
   </div>



The KLIP Pipeline
-----------------

We will now go through each major stage of the KLIP algorithm.




Step 0: Original Data
------------------------------

The first step involves loading your science data, which typically consists of a sequence of images (a data cube) and their corresponding parallactic angles. 
The data might also undergo initial pre-processing like bad pixel correction, flat-fielding, or cosmic ray removal (though these are often done before input to ``torchklip``).


.. raw:: html

    <div id="controls" style="width:600px; margin:20px auto;">

    <!-- label on its own line -->
    <div style="text-align:center; color:var(--ctrl-text); font-size:14px; margin-bottom:8px;">
      Time Frame:
    </div>

    <!-- slider + bubble in a relative container -->
    
    <div style="position:relative; height:40px;">
      <input
        id="param-slider"
        type="range"
        min="1" max="61" value="1"
        style="--pct:0%;"
      />
      
      <span
        id="param-label"
        style="
          position:absolute;
          top:25px;           /* pushed down below the label line */
          left:0;
          transform:translateX(-50%);
          background:var(--ctrl-bg);
          color:var(--ctrl-text);
          padding:2px 6px;
          border-radius:4px;
          font-size:12px;
          pointer-events:none;
        "
      >1</span>
    </div>
    <div style="display:flex; justify-content:space-between; color:var(--ctrl-text); font-size:12px; margin-top:8px;">
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

      #controls {
        color: #fff;
        font-family: sans-serif;
        font-size: 14px;
    }
    /* style each label as an inline-flex checkbox + text block */
    #controls label {
        display: inline-flex;
        align-items: center;
        margin-right: 24px;
        cursor: pointer;
        user-select: none;
    }
    /* enlarge the native checkbox, add a nice accent color */
    #controls input[type="checkbox"] {
        width: 18px;
        height: 18px;
        margin-right: 6px;
        accent-color: #5369FF;
        cursor: pointer;
    }
    /* on hover, slightly brighten the label text */
    #controls label:hover {
        color: #e0e0e0;
    }

    #pixel-tooltip {
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
  <div id="controls" style="text-align:center; margin-bottom:10px; color:var(--ctrl-text);">
    <label for="log-scale">
        <input id="log-scale" type="checkbox" checked>
        Log scale
    </label>
    <label for="show-iwa">
        <input id="show-iwa" type="checkbox" checked>
        Show IWA
    </label>
    <label for="show-center">
        <input id="show-center" type="checkbox" checked>
        Show center
    </label>
    </div>

    <div id="plot-container" style="position: relative; display: flex; justify-content: center; margin-left:60px">
        <!-- SVG will be injected here by the script -->
        <div id="pixel-tooltip"></div>
    </div>

Step 1: Input Data Preparation
------------------------------

For KLIP, the data is often flattened for localized processing.

.. raw:: html

    <div id="step1"></div>


The data is mean-subtracted and NaNs are replaced with zeros.
The 3D datacube (frames x height x width) is flattened to a 2D matrix where each row represents a frame.

.. code-block:: python

    processed = datacube.mean_subtract_()
    data = torch.nan_to_num(processed.tensor, nan=0.0)
    ref_flat = data.view(nk, -1)


Step 2: Computing KL Basis 
---------------------------
From a library of reference frames (selected based on criteria like sufficient sky rotation to avoid self-subtraction), KLIP computes a set of basis vectors. These Karhunen-Loève (KL) modes, or eigenimages, represent the principal components of the speckle noise and stellar PSF structure found in the reference frames for the chosen region (e.g., an annulus).

The algorithm computes a set of orthogonal basis vectors (principal components) that 
efficiently represent the variability in the reference

This is done using SVD (Singular Value Decomposition), PCA (Principal Component Analysis), or eigendecomposition methods. 
The basis vectors (Z_KL) represent different modes of PSF variation.

In the process, the basis vectors (Z_KL) are flattened to match the shape of the reference frames.
Here we reshape the basis vectors to match the original image dimensions for visualization.

.. raw:: html 

   <div id="step2" style="text-align: center;">
     <img src="../_static/images/step2.png" alt="KL Basis ">
   </div>


.. raw:: html

    <div id="controls" style="width:600px; margin:20px auto;">

    <!-- label on its own line -->
    <div style="text-align:center; font-size:14px; margin-bottom:8px; color:var(--ctrl-text);">
      Principal Component:
    </div>

    <!-- slider + bubble in a relative container -->
    <div style="position:relative; height:40px;">
      <input
        id="pca-slider"
        type="range"
        min="1" max="61" value="1"
        style="--pct:0%;"
      />
      <span
        id="pca-label"
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

      #pca-tooltip {
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

    <div id="pca-container" style="position:relative; display: flex; justify-content: center">
        <div id="pca-tooltip"></div>
    </div>


.. code-block:: python
    
    Z_KL = basis_fn(ref_flat, K_max)

Step 3: Projection onto KL Basis
--------------------------------

Each image frame is projected onto the KL basis to determine the contribution of each KL mode to that frame.

These projection coefficients quantify how much of each principal component is present in each image frame. 
Mathematically, this is a matrix multiplication between the reference images and the KL basis.

.. code-block:: python
    
    proj = ref_flat @ Z_KL

These projection coefficients quantify how much of each principal component is present in each image frame. 
Mathematically, this is a matrix multiplication between the reference images and the KL basis.

Step 4: Reconstructing PSF Model 
--------------------------------

The KL basis and projection coefficients are used to reconstruct a model of the stellar PSF for each frame.

By multiplying the projection coefficients back with the transposed KL basis, we get the reconstructed PSF pattern. 
This represents our best estimate of the stellar light contribution that we want to remove.

.. code-block:: python
    
    ihat = proj @ Z_KL.T

Step 5: Subtracting PSF Model 
-----------------------------

The reconstructed PSF model is subtracted from the original data to reveal potential exoplanet signals.

.. raw:: html

    <div id="pcsf-controls">
      <label>
      K value:
      <input id="pcsf-k-slider" type="range" list="k-ticks" min="1" max="7" step="1" value="1">
      <datalist id="k-ticks">
          <option value="1" label="5"></option>
          <option value="2" label="10"></option>
          <option value="3" label="20"></option>
          <option value="4" label="30"></option>
          <option value="5" label="40"></option>
          <option value="6" label="50"></option>
          <option value="7" label="60"></option>
      </datalist>
      <span id="pcsf-k-label">5</span>
      </label>
      <label>
      Time frame:
      <input id="pcsf-t-slider" type="range" list="t-ticks" min="1" max="7" step="1" value="1">
      <datalist id="t-ticks">
          <option value="1" label="5"></option>
          <option value="2" label="10"></option>
          <option value="3" label="20"></option>
          <option value="4" label="30"></option>
          <option value="5" label="40"></option>
          <option value="6" label="50"></option>
          <option value="7" label="60"></option>
      </datalist>
      <span id="pcsf-t-label">5</span>
      </label>
    </div>

    <!-- three panels + operators + tooltip container -->
    <div id="pcsf-plots">
        <div class="pcsf-panel">
        <h4>Original Image</h4>
        <div class="pcsf-image-container">
            <canvas id="pcsf-original" width="200" height="200" style="position: relative;"></canvas>
        </div>
        </div>
        <div class="pcsf-op">−</div>
        <div class="pcsf-panel">
        <h4>Reconstructed PSF</h4>
        <div class="pcsf-image-container">
            <canvas id="pcsf-psf" width="200" height="200" style="position: relative;"></canvas>
        </div>
        </div>
        <div class="pcsf-op">=</div>
        <div class="pcsf-panel">
        <h4>Residual Image</h4>
        <div class="pcsf-image-container">
            <canvas id="pcsf-residual" width="200" height="200" style="position: relative;"></canvas>
        </div>
        </div>
        <div id="pcsf-tooltip"></div>
        <div id="pcsf-value-overlay"></div>
    </div>
    <style>
      /* ── Controls ───────────────────────────────────────────────────────────── */
      #pcsf-controls {
        text-align: center;
        margin: 20px auto;
        width: 600px;
        color: var(--ctrl-text, #fff);
      }
      #pcsf-controls label {
        margin: 0 16px;
        user-select: none;
      }
      #pcsf-controls input[type="range"] { width: 200px; vertical-align: middle; }
      #pcsf-controls span { margin-left: 8px; font-weight: bold; }

      /* ── Layout of panels + operators ────────────────────────────────────────── */
      #pcsf-plots {
        display: flex;
        flex-wrap: nowrap;
        align-items: center;
        justify-content: center;
        gap: 16px;
        margin-top: 24px;
        position: relative;
      }
      .pcsf-panel {
        text-align: center;
      }
      .pcsf-image-container {
        display: flex;
        flex-direction: row;
        align-items: center;
      }
      .pcsf-op {
        width: 32px;
        text-align: center;
        font-size: 24px;
        color: var(--ctrl-text, #fff);
      }

      /* ── Tooltip ─────────────────────────────────────────────────────────────── */
      #pcsf-tooltip {
        position: absolute;
        pointer-events: none;
        background: rgba(0,0,0,0.7);
        color: #fff;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 12px;
        opacity: 0;
        transition: opacity 0.1s;
        z-index: 10;
      }
      #pcsf-value-overlay {
        display: none;
        position: absolute;
        pointer-events: none;
        z-index: 5;
      }
      </style>



.. code-block:: python
    
    residual_flat = ref_flat - ihat
    residuals = residual_flat.view(nk, ny, nx)

This subtraction removes the starlight and quasi-static speckle noise, ideally leaving behind any planetary signals and random noise. 
The key insight of KLIP is that the planet signal does not project strongly onto the KL basis derived from reference PSFs.



Step 6: Derotation and Combination
----------------------------------

Each residual frame is rotated to align the potential planetary signals, then combined to enhance the signal-to-noise ratio.

.. raw:: html 

   <div id="step6" style="text-align: center;">
     <img src="../_static/images/step6.png" alt="Derotation and Combination">
   </div>

Since the telescope frames were taken at different rotation angles, we need to counter-rotate the residuals to align any planetary signals. 
The aligned frames are then combined using mean or median to increase the signal-to-noise ratio of the planet relative to the random noise.

.. code-block:: python
    
    derot = derotate_cube(residuals, angles, batch=True)
    result = combine_cube(derot, statistic='mean')


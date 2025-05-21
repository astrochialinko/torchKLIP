.. torchklip documentation master file, created by
   sphinx-quickstart on Wed Apr 23 16:17:59 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TorchKLIP's documentation!
=====================================

TorchKLIP is a PyTorch-based implementation of the Karhunen-Loève Image Processing (KLIP) algorithm for high-contrast exoplanet direct imaging.

It provides efficient and GPU-accelerated processing of astronomical data for applications such as exoplanet detection and characterization.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   install
   tutorials
   documentation
   about
   release



Quick Start
-----------

.. code-block:: python

   import torchklip as tkl
   import numpy as np
    
   # Load your astronomical data
   data = tkl.load_data("your_data.fits")
   angles = np.load("your_angles.npy")
    
   # Preprocess the data with masking
   tensordata = tkl.DataTensor(cube, center = [50, 501], IWA=4) 
    
   # Run the KLIP algorithm
   device = "cuda" if torch.cuda.is_available() else "cpu"   
   tklip = tkl.TorchKLIP(tensordata, angles, device=device)
    
   # Process with 10 KL modes
   result = tklip.klip_and_derotate(K_klip=10)


Indices
-------

* :ref:`genindex`
* :ref:`search`
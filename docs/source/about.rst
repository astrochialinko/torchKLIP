About
=====

torchKLIP is a PyTorch implementation of the Karhunen–Loève Image Projection
(KLIP) algorithm for point spread function (PSF) subtraction in high-contrast
imaging of exoplanets and circumstellar disks.

Building on PyTorch lets the same code run on CPUs and GPUs, and makes it easy
to combine PSF subtraction with other machine-learning tools. torchKLIP
supports angular differential imaging (ADI), reference differential imaging
(RDI) and a hybrid ADI+RDI mode.

Background
----------

KLIP was introduced by Soummer, Pueyo & Larkin (2012, ApJL, 755, L28). It
builds an orthogonal basis of PSF modes from a reference library using
principal component analysis, projects each science frame onto the first K
modes to model the stellar PSF, and subtracts that model to reveal faint
companions.

Citation
--------

If you use torchKLIP in your research, please cite
`Ko et al. (2024) <https://doi.org/10.1117/12.3027407>`_
(`arXiv:2409.16466 <https://arxiv.org/abs/2409.16466>`_), as well as
Soummer et al. (2012) for the KLIP algorithm.

.. code-block:: bibtex

   @inproceedings{10.1117/12.3027407,
     author    = {Chia-Lin Ko and Ewan S. Douglas and Justin Hom},
     title     = {{A PyTorch benchmark for high-contrast imaging post processing}},
     booktitle = {Applications of Machine Learning 2024},
     volume    = {13138},
     pages     = {1313811},
     publisher = {SPIE},
     year      = {2024},
     doi       = {10.1117/12.3027407}
   }

Acknowledgments
---------------

We thank the `Code/Astro Workshop <https://semaphorep.github.io/codeastro/>`_
for training in the development of open-source software packages.

License and links
-----------------

torchKLIP is released under the MIT License.

* Source code: https://github.com/astrochialinko/torchKLIP
* Package: https://pypi.org/project/torchklip/
* Report issues: https://github.com/astrochialinko/torchKLIP/issues
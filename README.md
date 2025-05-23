# torchKLIP

[![arXiv](https://img.shields.io/badge/arXiv-2409.16466-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2409.16466)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?logo=open-source-initiative&logoColor=white)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://img.shields.io/badge/docs-latest%20build-brightgreen?logo=read-the-docs&logoColor=white)](https://torchklip.readthedocs.io/en/latest/?badge=latest)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)](https://pypi.org/project/torchklip/)
[![PyPI Status](https://img.shields.io/pypi/v/torchklip.svg?logo=pypi&logoColor=white&color=purple)](https://pypi.org/project/torchklip/)

torchKLIP is a PyTorch implementation of the PCA-based Karhunen–Loève Image Projection (KLIP) algorithm for point spread function (PSF) subtraction in the direct imaging of exoplanets.

## Set-Up

Choose one of the following methods depending on your workflow.

**Recommended:** Create an isolated environment first to avoid dependency conflicts. For example, using conda:

```bash
conda create -n torchklip python=3.11
conda activate torchklip
```

### Install from PyPI

Install the stable version on [PyPI](https://pypi.org/project/torchklip/) with:

```bash
pip install torchklip
```

### Install from GitHub

Install the latest code directly from the repository:

```bash
pip install git+ssh://git@github.com/astrochialinko/torchKLIP.git
```

### Development Mode

If you plan to contribute or customize the code, fork the repository on GitHub and clone your fork:

```bash
# Fork on GitHub, then:
git clone git@github.com:<your-github-username>/torchKLIP.git
cd torchKLIP
```

If you're hacking locally without contributing back, you can clone the main repo directly:

```bash
git clone git@github.com:astrochialinko/torchKLIP.git
cd torchKLIP
```

Install in editable mode using `pip` with `pyproject.toml`

```bash
pip install -e .
```

### Jupyter Notebook

If you plan to use Jupyter Notebook, as recommended, , first install the Jupyter kernel package

```bash
pip install ipykernel
```

Then register a kernel for this environment

```bash
python3 -m ipykernel install --user --name "torchklip" --display-name "torchklip"
```

## Documentation

The full documentation, including detailed API references, installation steps, example datasets, and tutorials, is available on [Read the Docs](https://torchklip.readthedocs.io).

## Quick Start

To quickly test torchKLIP, follow these steps:

1. (Optional) Make sure the Jupyter Notebook interface is installed:
   ```bash
   pip install notebook  # or: conda install notebook
   ```
2. Launch the [example notebook](notebooks/tutorial_betaPic.ipynb):
   ```bash
   jupyter notebook notebooks/tutorial_betaPic.ipynb
   ```
3. In the Jupyter interface, select the `torchklip` kernel you created earlier.

This notebook walks through a KLIP-based PSF subtraction example using the Beta Pictoris dataset and demonstrates key features of the package.

# Project Structure

```
.
├── LICENSE                          # License information
├── README.md                        # This overview file
├── data/                            # (Optional) Example data or staging area
├── docs/                            # Documentation sources (Sphinx/ReadTheDocs)
├── logs/                            # Runtime logs and profiler output
├── notebooks/                       # Jupyter notebooks for tutorials and experiments
│   └── tutorial_betaPic.ipynb       # Demo notebook on Beta Pictoris dataset
├── pyproject.toml                   # Build configuration and metadata
├── src/torchklip/                   # Core package source code
│   ├── __init__.py                  # Package entry point
│   ├── algos/                       # KLIP algorithm implementations
│   │   ├── __init__.py
│   │   └── klip/                    # SVD, Eign, and PCA solver for KLIP
│   │       ├── __init__.py
│   │       ├── eigh.py              # Eigensolver using SciPy
│   │       ├── klip_base.py         # Base KLIP class and common logic
│   │       ├── pca.py               # PCA-based projection implementation
│   │       └── svd.py               # SVD-based projection variant
│   ├── config.py                    # Configuration management and defaults
│   ├── dataproc/                    # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── data_loader.py           # Dataloader for FITS or image stacks
│   │   └── data_preprocessor.py     # Centering, scaling, and cropping routines
│   └── utils/                       # Utility functions and helpers
│       ├── __init__.py
│       ├── image_plot.py            # Plotting routines for results
│       ├── logging_utils.py         # Setup for structured logging
│       ├── metrics_renderer.py      # metric computations
│       ├── profiler.py              # Timing and performance profiling
│       └── snr.py                   # Signal-to-noise ratio calculations
└── tests/                           # Unit tests
```

# Citation

If you use torchKLIP in your research, please cite [Ko et al. (2024)](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13138/1313811/A-PyTorch-benchmark-for-high-contrast-imaging-post-processing/10.1117/12.3027407.short)

> C.-L. Ko, E. S. Douglas, and J. Hom. A pytorch benchmark for high-
> contrast imaging post processing. In Applications of Machine Learning
> 2024, vol. 13138, pp. 229–236. SPIE, 2024.

BibTeX

```latex
@inproceedings{10.1117/12.3027407,
author = {Chia-Lin Ko and Ewan S. Douglas and Justin Hom},
title = {{A PyTorch benchmark for high-contrast imaging post processing}},
volume = {13138},
booktitle = {Applications of Machine Learning 2024},
organization = {International Society for Optics and Photonics},
publisher = {SPIE},
pages = {1313811},
year = {2024},
doi = {10.1117/12.3027407},
URL = {https://doi.org/10.1117/12.3027407}}
```

# Acknowledgments

Thanks the [Code/Astro Workshop](https://semaphorep.github.io/codeastro/) for providing valuable training in the development of open-source software packages

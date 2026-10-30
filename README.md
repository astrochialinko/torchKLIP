# torchKLIP

[![arXiv](https://img.shields.io/badge/arXiv-2409.16466-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2409.16466)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.13-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?logo=open-source-initiative&logoColor=white)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://img.shields.io/readthedocs/torchklip?logo=read-the-docs&logoColor=white)](https://torchklip.readthedocs.io/en/latest/?badge=latest)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://pypi.org/project/torchklip/)
[![PyPI Status](https://img.shields.io/pypi/v/torchklip.svg?logo=pypi&logoColor=white&color=purple)](https://pypi.org/project/torchklip/)
[![Tests](https://img.shields.io/github/actions/workflow/status/astrochialinko/torchKLIP/tests.yml?branch=main&logo=github&logoColor=white&label=tests)](https://github.com/astrochialinko/torchKLIP/actions/workflows/tests.yml)

torchKLIP is a PyTorch implementation of the PCA-based Karhunen–Loève Image Projection (KLIP) algorithm for point spread function (PSF) subtraction in the direct imaging of exoplanets.

## Features

- **ADI**: full-frame angular differential imaging using the science frames themselves
- **RDI**: reference differential imaging with a separate reference-star library
- **Hybrid ADI+RDI**: combines reference frames with science frames taken at
  sufficiently different rotation angles
- Three KL basis solvers (`svd`, `pca`, `eigh`), with CPU and GPU support via PyTorch
- Diagnostics: KL eigenimages, PSF models, residuals and projection coefficients
- Multi-planet SNR measurement with keep-out zones around other planets
- FITS output helpers and plotting utilities

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
pip install git+https://github.com/astrochialinko/torchKLIP.git
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

If you plan to use Jupyter Notebook, as recommended, first install the Jupyter kernel package

```bash
pip install ipykernel
```

Then register a kernel for this environment

```bash
python3 -m ipykernel install --user --name "torchklip" --display-name "torchklip"
```

## Documentation

The full documentation, including detailed API references, installation steps, example datasets, and tutorials, is available on [Read the Docs](https://torchklip.readthedocs.io). See the [CHANGELOG](https://github.com/astrochialinko/torchKLIP/blob/main/CHANGELOG.md) for release notes.

## Quick Start

To quickly test torchKLIP, follow these steps:

1. (Optional) Make sure the Jupyter Notebook interface is installed:
   ```bash
   pip install notebook  # or: conda install notebook
   ```
2. Launch the [example notebook](https://github.com/astrochialinko/torchKLIP/blob/main/notebooks/tutorial_betaPic.ipynb):
   ```bash
   jupyter notebook notebooks/tutorial_betaPic.ipynb
   ```
3. In the Jupyter interface, select the `torchklip` kernel you created earlier.

This notebook walks through a KLIP-based PSF subtraction example using the Beta Pictoris dataset and demonstrates key features of the package.

## Usage

```python
import torch
from torchklip import TorchKLIP, DataTensor

sci = DataTensor(sci_cube)        # (N, H, W) science frames
ref = DataTensor(ref_cube)        # (M, H, W) reference-star frames
angles = torch.tensor(angles_deg) # (N,) derotation angles in degrees

klip = TorchKLIP(sci, angles, ref_data=ref)
K = [5, 10, 20]

adi    = klip.klip_and_derotate(K, mode="adi")
rdi    = klip.klip_and_derotate(K, mode="rdi")
hybrid = klip.klip_and_derotate(K, mode="hybrid", annular_exclusion=10.0)
```

Each call returns a `(len(K), H, W)` tensor of derotated, combined residuals.
See the [documentation](https://torchklip.readthedocs.io) for all options.

## Citation

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

## Acknowledgments

We thank the [Code/Astro Workshop](https://semaphorep.github.io/codeastro/) for providing valuable training in the development of open-source software packages.

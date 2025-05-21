# torchKLIP

torchKLIP is a PyTorch implementation of the PCA-based Karhunen–Loève Image Projection (KLIP) algorithm for point spread function (PSF) subtraction in the direct imaging of exoplanets.

## Set-Up

Choose one of the following methods depending on your workflow.

**Recommended:** Create an isolated environment first to avoid dependency conflicts. For example, using conda:

```bash
conda create -n torchklip python=3.11
conda activate torchklip
```

### Install from PyPI

Install the stable version with:

```bash
pip install torchKLIP
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

Install in editable mode:

```bash
pip install -r requirements.txt
python setup.py develop
```

### Jupyter Notebook

If you plan to use Jupyter Notebook, as recommended, install a Jupyter kernel from this environment:

```bash
python3 -m ipykernel install --user --name "torchklip" --display-name "torchklip"
```

## Quick Start

Open the [example notebook](notebooks/tutorial_betaPic.ipynb) and choose the `torchklip` kernel:

```bash
jupyter notebook notebooks/tutorial_betaPic.ipynb
```

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
> 2024, vol. 13138, pp. 229–236. SPIE, 2024. [doi:10.1117/12.3027407]

# Acknowledgments

- Thanks the [Code/Astro Workshop](https://semaphorep.github.io/codeastro/) for providing valuable training in the development of open-source software packages

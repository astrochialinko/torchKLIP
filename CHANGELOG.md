# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-10-30

### Added
- Reference differential imaging (RDI): pass a reference cube as `ref_data`
  and use `mode="rdi"`.
- Hybrid ADI+RDI mode (`mode="hybrid"`) with PA batching and angular exclusion.
- `select_reference_frames` for correlation-based reference frame selection.
- Optional reference flux normalisation (`normalize_ref`).
- `KLIPDiagnostics` exposing KL eigenimages, PSF models, residuals and
  projection coefficients via `return_diagnostics=True`.
- Multi-planet SNR with distance-based keep-out zones, noise aperture helpers
  and noise histograms.
- New modules: `torchklip.io` (FITS output) and `torchklip.plotting`.
- `torchklip.__version__`.
- Continuous integration on Python 3.11 and 3.12.

### Changed
- `compute_basis_eigh` computes in float64 for numerical stability and raises
  `ValueError` when `K_max` exceeds the number of library frames.

### Fixed
- Added the missing `psutil` dependency, which made `import torchklip` fail in
  clean environments.

## [0.1.0]

- Initial release.

[0.2.0]: https://github.com/astrochialinko/torchKLIP/compare/v0.1...v0.2.0
[0.1.0]: https://github.com/astrochialinko/torchKLIP/releases/tag/v0.1

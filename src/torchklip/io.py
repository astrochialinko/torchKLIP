# torchklip/io.py
"""FITS I/O helpers for KLIP pipeline outputs."""

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Union

import numpy as np
from astropy.io import fits

from .utils.logging_utils import get_logger

logger = get_logger(__name__.split(".")[-1])

__all__ = ["save_fits_outputs"]


def save_fits_outputs(
    result_np: np.ndarray,
    kl_basis_np: np.ndarray,
    k_list: List[int],
    output_dir: Union[str, Path],
    label: str,
    method: str,
    mode: str,
    bunit: str = "adu",
) -> None:
    """Save the collapsed residual cube and KL eigenimage cube as FITS files.

    Writes three kinds of file per call:

    * ``{label}_residuals_cube.fits`` — 3-D ``(n_K, H, W)`` cube containing
      the combined residual for every K in *k_list*.
    * ``{label}_combined_K{k}.fits`` — individual 2-D ``(H, W)`` image per K.
    * ``{label}_kl_basis.fits`` — 3-D ``(K_max, H, W)`` KL eigenimage cube
      (last PA batch for hybrid; global basis for ADI/RDI).

    Array dimensions (``NAXIS*``) are written by astropy from the data.

    Args:
        result_np   : ``(n_K, H, W)`` numpy array of combined residuals.
        kl_basis_np : ``(K_max, H, W)`` numpy array of KL eigenimages.
        k_list      : Sorted list of K values (must match axis-0 of *result_np*).
        output_dir  : Directory to write FITS files into; created if missing.
        label       : Short label prefix for all filenames (e.g. ``"os11_adi"``).
        method      : Basis method string (``"svd"``, ``"pca"``, or ``"eigh"``).
        mode        : Processing mode string (``"adi"``, ``"rdi"``, or ``"hybrid"``).
        bunit       : Pixel unit written to ``BUNIT`` (default ``"adu"``).  Set
                      this to match the input data, e.g. after flux normalisation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    k_str = ",".join(str(k) for k in k_list)
    K_max = kl_basis_np.shape[0]
    n_K = result_np.shape[0]

    def _common_header() -> fits.Header:
        hdr = fits.Header()
        hdr["METHOD"] = (method.upper(), "Basis computation method")
        hdr["MODE"] = (mode.upper(), "KLIP processing mode")
        hdr["DATE"] = (date_str, "UTC creation timestamp")
        return hdr

    # 3-D residual cube (n_K, H, W)
    hdr_cube = _common_header()
    hdr_cube["KLIST"] = (k_str, "K values (comma-separated)")
    hdr_cube["BUNIT"] = (bunit, "Pixel unit")
    hdr_cube.add_comment("KLIP PSF-subtracted & derotated residual cube")
    hdr_cube.add_comment("Axis 3 (NAXIS3): K index; axes 1-2: spatial (x, y)")
    cube_path = output_dir / f"{label}_residuals_cube.fits"
    fits.writeto(cube_path, result_np.astype(np.float32), header=hdr_cube, overwrite=True)
    logger.info(f"Saved residual cube → {cube_path.name}  shape={result_np.shape}")

    # Individual 2-D images per K
    for k_idx, k in enumerate(k_list):
        hdr_2d = _common_header()
        hdr_2d["K_KLIP"] = (k, "Number of KL modes used")
        hdr_2d["KLIST"] = (k_str, "Full K list (comma-separated)")
        hdr_2d["BUNIT"] = (bunit, "Pixel unit")
        hdr_2d.add_comment(f"KLIP {mode.upper()} collapsed residual image (K={k})")
        fits.writeto(
            output_dir / f"{label}_combined_K{k}.fits",
            result_np[k_idx].astype(np.float32),
            header=hdr_2d,
            overwrite=True,
        )
    logger.info(f"Saved {n_K} per-K FITS images → {label}_combined_K*.fits")

    # KL eigenimage cube (K_max, H, W)
    hdr_kl = _common_header()
    hdr_kl["KMAX"] = (K_max, "Total KL modes computed")
    hdr_kl.add_comment("KL eigenimage cube (K_max, ny, nx)")
    hdr_kl.add_comment("Axis 3 (NAXIS3): KL mode index (1-based); axes 1-2: spatial")
    if mode == "hybrid":
        hdr_kl.add_comment("Hybrid mode: basis reflects the last PA batch only")
    kl_path = output_dir / f"{label}_kl_basis.fits"
    fits.writeto(kl_path, kl_basis_np.astype(np.float32), header=hdr_kl, overwrite=True)
    logger.info(f"Saved KL basis cube → {kl_path.name}  shape={kl_basis_np.shape}")
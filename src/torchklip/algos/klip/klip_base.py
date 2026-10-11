# File: src/torchklip/algos/klip/klip_base.py
# Modified: RDI/hybrid mask consistency, optional reference normalisation,
#           reference-frame selection, size validation.
import time
from dataclasses import dataclass
from typing import Union, List, Optional, Tuple

import torch
import numpy as np

# local modules
from ...dataproc.data_preprocessor import DataTensor
from ...utils.logging_utils import get_logger
from ...utils.profiler import PerformanceMonitor
from ...utils.metrics_renderer import bytes_to_human, format_time
from .svd import compute_basis_svd
from .pca import compute_basis_pca
from .eigh import compute_basis_eigh

# Get a logger specific to this module
logger = get_logger(__name__.split('.')[-1])

_basis_strategies = {
    "svd": compute_basis_svd,
    "pca": compute_basis_pca,
    "eigh": compute_basis_eigh,
}


@dataclass
class ResidualsWithIntermediates:
    """
    Container for residuals and KLIP intermediates.

    Attributes:
        residuals: Tensor of shape (n_k_values, nk, ny, nx).
        Z_KL: Tensor of shape (ny*nx, K_max), the KL basis.
        proj: Tensor of shape (nk, K_max), the projection coefficients.
        ihat: Tensor of shape (nk, ny*nx), the reconstructed frames from KL basis.
        residual_flat: Tensor of shape (nk, ny*nx), the flattened residuals.
    """
    residuals: torch.Tensor
    Z_KL: torch.Tensor
    proj: torch.Tensor
    ihat: torch.Tensor
    residual_flat: torch.Tensor

    def __iter__(self):
        """Enable tuple unpacking of the dataclass."""
        yield self.residuals
        yield self.Z_KL
        yield self.proj
        yield self.ihat
        yield self.residual_flat


@dataclass
class KLIPDiagnostics:
    """Diagnostic arrays from KLIP PSF subtraction.

    All spatial arrays are returned in image shape ``(ny, nx)`` rather than
    flattened, making them directly plottable.

    Attributes:
        kl_basis: KL-mode eigenimages of shape ``(K_max, ny, nx)``.
            ``kl_basis[i]`` is the (i+1)-th KL mode image.
        mean_frame: Temporal mean of the input datacube, shape ``(ny, nx)``.
            This is the average frame over all exposures before KLIP subtraction
            and represents the common stellar PSF structure.
        psf_estimates: Reconstructed stellar PSF per K value,
            shape ``(n_K, nk, ny, nx)``.  ``psf_estimates[i]`` is the PSF
            model built from ``K_list[i]`` KL modes for every input frame.
        residual_frames: Pre-collapse PSF-subtracted frames per K value,
            shape ``(n_K, nk, ny, nx)``.  These are the residuals before
            derotation and temporal combination.
        coefficients: Projection weights of each frame onto all ``K_max`` KL
            modes, shape ``(nk, K_max)``.  Slice ``coefficients[:, :k]`` to
            retrieve weights for a specific ``k``.  In hybrid mode the basis
            varies per PA batch, so coefficients reflect each frame's projection
            onto its own batch basis (zero-padded to ``K_max`` when the
            effective library was smaller).
        K_list: Sorted list of K values used in this run.
    """
    kl_basis: torch.Tensor        # (K_max, ny, nx)
    mean_frame: torch.Tensor      # (ny, nx)
    psf_estimates: torch.Tensor   # (n_K, nk, ny, nx)
    residual_frames: torch.Tensor # (n_K, nk, ny, nx)
    coefficients: torch.Tensor    # (nk, K_max)
    K_list: List[int]


def _normalize_and_validate_K(K_klip: Union[int, List[int]], nk: int, logger) -> List[int]:
    """Normalizes K_klip to a sorted list of positive integers <= nk with validation.

    Args:
        K_klip: Integer or list of integers specifying KL mode counts.
        nk: Number of frames in the datacube.
        logger: Logger instance for recording status and errors.

    Returns:
        List of validated, sorted KL mode counts.

    Raises:
        ValueError: If K_klip list is empty, contains non-positive integers,
            or requests more modes than frames available.
        TypeError: If any entry in K_klip is not an integer.
    """
    logger.debug(f"Normalizing K_klip input: {K_klip!r}")

    # Normalize
    K_list = [K_klip] if isinstance(K_klip, int) else list(K_klip)

    # Validate
    if not K_list:
        logger.error("Empty K_klip list")
        raise ValueError("K_klip list cannot be empty.")

    if any(not isinstance(k, int) for k in K_list):
        logger.error("Non-integer in K_klip list")
        raise TypeError("All entries in K_klip must be integers.")

    if any(k < 1 for k in K_list):
        logger.error("Non-positive integer in K_klip list")
        raise ValueError("All KL mode counts must be >= 1.")

    K_list = sorted(K_list)
    K_max = K_list[-1]

    if K_max > nk:
        logger.error(
            f"Requested K_max={K_max} > number of frames nk={nk}"
        )
        raise ValueError(
            f"Cannot request {K_max} modes; datacube only has {nk} frames.")
    logger.debug(f"Using K_list={K_list}")

    return K_list


def compute_residuals(datacube: DataTensor,
                      K_klip: Union[int, List[int]],
                      method: str = "svd",
                      store_intermediates: bool = False,
                      return_diagnostics: bool = False,
                      ref_datacube: Optional[DataTensor] = None,
                      normalize_ref: bool = False,
                      ) -> Union[torch.Tensor,
                                 ResidualsWithIntermediates,
                                 Tuple[torch.Tensor, "KLIPDiagnostics"]]:
    """
    Compute PSF-subtracted residual cube for given KL modes.

    This function implements the core KLIP (Karhunen-Loeve Image Processing) algorithm
    to perform PSF subtraction. It computes the KL basis, projects the input frames,
    and returns the residual frames after subtracting the reconstructions.

    When ``ref_datacube`` is provided the function operates in **RDI mode**: the KL
    basis is built from the reference library and the projection/subtraction is
    applied to the science (target) ``datacube``.  K is then bounded by the number
    of reference frames rather than target frames.

    Args:
        datacube: DataTensor containing the science (target) frames.
        K_klip: Integer or list of integers specifying KL mode counts.
        method: Basis computation method, one of 'svd', 'pca', or 'eigh'.
        store_intermediates: If True, return flat intermediate arrays in a
            :class:`ResidualsWithIntermediates` (only for K_max).
            Ignored when ``return_diagnostics=True``.
        return_diagnostics: If True, return a ``(residuals, KLIPDiagnostics)``
            tuple containing image-shaped diagnostic arrays for every K value.
            Takes precedence over ``store_intermediates``.
        ref_datacube: Optional reference-star DataTensor for RDI.  When
            supplied the KL basis is derived from these frames; ``datacube``
            (science) is only used for projection and residual computation.
            Its image size must match ``datacube`` and its masked (NaN) pixels
            should be a superset of the science mask (``TorchKLIP`` ensures
            this automatically).
        normalize_ref: RDI only.  If True, every reference frame is scaled to
            unit L2-norm (after mean subtraction) before the KL basis is
            built, so bright reference frames do not dominate the covariance.
            Projection of the science frames is unaffected.  Default False
            (identical to the previous behaviour).

    Note:
        ``datacube`` and ``ref_datacube`` are mean-subtracted **in place**.
        ``TorchKLIP`` passes private copies, so this only matters when
        calling :func:`compute_residuals` directly.

    Returns:
        * Default (both flags False): Tensor of shape ``(n_k_values, nk, ny, nx)``.
        * ``store_intermediates=True``: :class:`ResidualsWithIntermediates` with
          attributes ``(residuals, Z_KL, proj, ihat, residual_flat)``.
        * ``return_diagnostics=True``: ``(residuals, KLIPDiagnostics)`` where
          *residuals* has shape ``(n_k_values, nk, ny, nx)``.

    Raises:
        ValueError: If K_klip list is empty, contains non-positive integers,
            or requests more modes than frames available.
        TypeError: If any entry in K_klip is not an integer.
        ValueError: If method is not in the supported strategies.
    """
    logger.info("Starting compute_residuals")
    nk, ny, nx = datacube.nk, datacube.ny, datacube.nx

    # RDI: the reference must live on the same pixel grid as the science cube.
    if ref_datacube is not None and \
            tuple(ref_datacube.shape[1:]) != (ny, nx):
        raise ValueError(
            f"ref_datacube image size {tuple(ref_datacube.shape[1:])} != "
            f"science image size {(ny, nx)}")

    # RDI: K is bounded by the reference library size, not the target cube.
    nk_basis = ref_datacube.nk if ref_datacube is not None else nk

    # Normalize and validate K_klip
    K_list = _normalize_and_validate_K(K_klip, nk_basis, logger)
    K_max = K_list[-1]

    # Capture temporal mean of TARGET before in-place mean subtraction
    if return_diagnostics:
        mean_frame = torch.nanmean(datacube.tensor, dim=0)  # (ny, nx)

    # Prepare science data: mean‐subtract and replace NaNs
    logger.debug("Mean-subtracting datacube and replacing NaNs")
    processed = datacube.mean_subtract_()
    data = torch.nan_to_num(processed.tensor, nan=0.0)
    target_flat = data.view(nk, -1)  # shape: (nk, ny*nx)

    # Build the matrix used to compute the KL basis
    if ref_datacube is not None:
        logger.debug("RDI mode: building KL basis from reference library")
        ref_processed = ref_datacube.mean_subtract_()
        ref_data = torch.nan_to_num(ref_processed.tensor, nan=0.0)
        basis_input = ref_data.view(ref_datacube.nk, -1)  # (nk_ref, ny*nx)
        if normalize_ref:
            logger.debug("RDI: L2-normalising reference frames")
            basis_input = _normalize_library(basis_input)
    else:
        basis_input = target_flat  # ADI: self-reference

    # Compute KL basis
    logger.info(f"Computing KL basis with method={method!r}, K_max={K_max}")
    try:
        basis_fn = _basis_strategies[method]
    except KeyError:
        error_msg = f"Unknown method: {method!r}; must be one of {list(_basis_strategies)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    Z_KL = basis_fn(basis_input, K_max)  # shape: (ny*nx, K_max)
    logger.debug(f"Basis shape: {tuple(Z_KL.shape)}")

    # Alias so the projection loop below works identically for ADI and RDI
    ref_flat = target_flat

    # Always compute a 4D tensor for residuals (n_k_values, nk, ny, nx)
    residuals = torch.zeros(len(K_list), nk, ny, nx, device=data.device)
    if return_diagnostics:
        psf_estimates = torch.zeros(len(K_list), nk, ny, nx, device=data.device)
    coefficients_max: Optional[torch.Tensor] = None

    # Compute residuals for each K value
    for i, k in enumerate(K_list):
        logger.debug(f"Projecting and reconstructing for k={k}")

        # Compute projection and residuals
        proj_k = ref_flat @ Z_KL[:, :k]      # shape: (nk, k)
        ihat_k = proj_k @ Z_KL[:, :k].T      # shape: (nk, ny*nx)
        residual_flat_k = ref_flat - ihat_k  # shape: (nk, ny*nx)
        residuals[i] = residual_flat_k.view(nk, ny, nx)

        if return_diagnostics:
            psf_estimates[i] = ihat_k.view(nk, ny, nx)

        # Store intermediates for max K if requested
        if store_intermediates and k == K_max:
            proj_max = proj_k
            ihat_max = ihat_k
            residual_flat_max = residual_flat_k

        # Track full-K_max coefficients (last iteration = K_max since K_list is sorted)
        coefficients_max = proj_k

    # Restore NaNs from original
    # shape: (1, nk, ny, nx) broadcast to (n_k_values, nk, ny, nx)
    logger.debug("Restoring NaNs to residuals")
    nan_mask = torch.isnan(datacube.tensor).unsqueeze(0)
    residuals = residuals.masked_fill(nan_mask, float("nan"))

    logger.info("Finished compute_residuals")

    # return_diagnostics takes precedence
    if return_diagnostics:
        psf_estimates = psf_estimates.masked_fill(nan_mask, float("nan"))
        diag = KLIPDiagnostics(
            kl_basis=Z_KL.T.view(K_max, ny, nx),   # (K_max, ny, nx)
            mean_frame=mean_frame,                   # (ny, nx)
            psf_estimates=psf_estimates,             # (n_K, nk, ny, nx)
            residual_frames=residuals,               # (n_K, nk, ny, nx)
            coefficients=coefficients_max,           # (nk, K_max)
            K_list=K_list,
        )
        return residuals, diag

    if store_intermediates:
        return ResidualsWithIntermediates(
            residuals, Z_KL, proj_max, ihat_max, residual_flat_max
        )
    return residuals


# ---------------------------------------------------------------------------
# Hybrid (ADI + RDI) helpers
# ---------------------------------------------------------------------------

def _normalize_library(frames: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Normalize each frame (row) of a library matrix to unit L2-norm.

    This ensures that every frame contributes equal power to the SVD
    covariance matrix, preventing a large reference library from
    numerically dominating a small target subset.

    Frames that are entirely zero or masked (all NaN→0) have a near-zero
    norm; their norm is clamped to *eps* so no division-by-zero occurs,
    and they effectively stay zero-valued after normalization.

    Args:
        frames: ``(n_frames, n_pixels)`` float tensor (NaNs already replaced
            with 0 by the caller).
        eps: Minimum norm value; prevents division by zero for blank frames.

    Returns:
        Normalized ``(n_frames, n_pixels)`` tensor where each row has
        L2-norm ≤ 1 (exactly 1 for non-blank frames).
    """
    norms = torch.norm(frames, dim=1, keepdim=True)  # (n_frames, 1)
    norms = torch.clamp(norms, min=eps)
    return frames / norms

def _group_frames_by_pa(
    angles: torch.Tensor,
    pa_reuse_threshold: float = 0.5,
) -> List[List[int]]:
    """Group consecutive frames by PA proximity for KL-basis reuse.

    Frames are clustered so that the absolute PA drift from the batch's
    representative (first) frame stays below *pa_reuse_threshold*.  A new
    batch is opened whenever the drift meets or exceeds the threshold.

    In a typical ADI observation the parallactic angle varies smoothly, so
    consecutive frames naturally form compact groups.  Within one group the
    set of angularly-excluded frames changes by less than *pa_reuse_threshold*
    degrees, making a single shared KL basis a good approximation for every
    member.

    Args:
        angles: 1-D tensor of parallactic angles in degrees, length ``nk``.
        pa_reuse_threshold: Maximum PA change (degrees) from the batch
            representative before starting a new batch.  Default ``0.5`` deg.

    Returns:
        List of groups; each group is a sorted list of frame indices.
        Groups are non-overlapping and together cover all ``nk`` frames.
    """
    nk = len(angles)
    if nk == 0:
        return []
    groups: List[List[int]] = []
    current_group: List[int] = [0]
    ref_pa: float = angles[0].item()
    for i in range(1, nk):
        if abs(angles[i].item() - ref_pa) < pa_reuse_threshold:
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group = [i]
            ref_pa = angles[i].item()
    groups.append(current_group)
    return groups


def compute_residuals_hybrid(
    datacube: DataTensor,
    K_klip: Union[int, List[int]],
    angles: torch.Tensor,
    method: str = "svd",
    annular_exclusion: float = 0.0,
    pa_reuse_threshold: float = 0.5,
    return_diagnostics: bool = False,
    ref_datacube: Optional[DataTensor] = None,
    hybrid_weight: float = 1.0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, KLIPDiagnostics]]:
    """Compute PSF-subtracted residuals in ADI+RDI hybrid (``'hybrid'``) mode.

    For each PA batch the reference library is assembled as

    .. math::

        \\mathcal{L} = \\mathcal{R}_{\\text{ref}}
                       \\cup
                       \\bigl\\{ I_j : |\\text{PA}_j - \\text{PA}_{\\text{rep}}|
                                        > \\delta_{\\text{excl}} \\bigr\\}

    where :math:`\\text{PA}_{\\text{rep}}` is the parallactic angle of the first
    frame in the batch, :math:`\\delta_{\\text{excl}}` is *annular_exclusion*,
    and :math:`\\mathcal{R}_{\\text{ref}}` is the external reference library
    (empty when *ref_datacube* is ``None``).

    A single KL basis is shared by all frames whose PA lies within
    *pa_reuse_threshold* degrees of the batch representative, avoiding a full
    SVD recomputation for every individual frame.

    Args:
        datacube: Science (target) DataTensor of shape ``(nk, ny, nx)``.
        K_klip: Integer or list of integers specifying KL mode counts.
        angles: 1-D tensor of parallactic angles in degrees, length ``nk``.
            Must reside on the same device as ``datacube``.
        method: Basis computation strategy — ``'svd'``, ``'pca'``, or
            ``'eigh'``.  Defaults to ``'svd'``.
        annular_exclusion: Angular exclusion half-width in degrees.  Target
            frames whose PA separation from the batch representative is at most
            this value are excluded from the self-reference subset of the
            library.  ``0.0`` (default) excludes only frames with exactly the
            same PA as the representative.
        pa_reuse_threshold: PA change (degrees) that triggers a new SVD
            computation.  All frames within the same batch share a single KL
            basis.  Default ``0.5`` deg.
        return_diagnostics: If ``True``, return a
            ``(residuals, KLIPDiagnostics)`` tuple.  ``kl_basis`` and
            ``coefficients`` are taken from the **last** PA batch;
            ``mean_frame`` is computed before any mean subtraction.
        ref_datacube: Optional external reference-star DataTensor.  Its frames
            are always included in the library regardless of their PA.  When
            ``None`` the mode degrades to ADI-only with angular exclusion.
        hybrid_weight: Multiplicative scale applied to the **normalized**
            target frames before concatenating them with the reference
            library for SVD.  Default ``1.0`` (equal amplitude).  Setting
            this above ``1.0`` upweights the self-reference (ADI) component
            so that a large reference library does not dominate the KL
            basis.  For example, ``hybrid_weight=3.0`` gives the target
            frames roughly 3× the per-frame amplitude of reference frames
            in the SVD covariance matrix.

            Both the reference and target libraries are per-frame L2
            normalized before this weight is applied, so the effect is
            independent of the raw flux scale of each dataset.

    Returns:
        * Default: Tensor ``(n_K, nk, ny, nx)`` of PSF-subtracted residuals.
        * ``return_diagnostics=True``: ``(residuals, KLIPDiagnostics)`` where
          *residuals* has the shape above.

    Raises:
        ValueError: If K_klip is invalid, or *method* is unrecognised, or
            ``K_max`` exceeds the maximum possible library size.
    """
    logger.info("Starting compute_residuals_hybrid")
    nk, ny, nx = datacube.nk, datacube.ny, datacube.nx
    n_ref = ref_datacube.nk if ref_datacube is not None else 0
    # Maximum possible library = all reference frames + all target frames
    max_lib_size = n_ref + nk

    K_list = _normalize_and_validate_K(K_klip, max_lib_size, logger)
    K_max = K_list[-1]

    # Capture temporal mean before in-place mean subtraction
    if return_diagnostics:
        mean_frame = torch.nanmean(datacube.tensor, dim=0)  # (ny, nx)

    # Prepare science frames: mean-subtract in-place, then replace NaNs with 0
    logger.debug("Hybrid: mean-subtracting science datacube")
    processed = datacube.mean_subtract_()
    data = torch.nan_to_num(processed.tensor, nan=0.0)
    target_flat = data.view(nk, -1)  # (nk, ny*nx)

    # Prepare reference frames when provided
    ref_flat: Optional[torch.Tensor] = None
    ref_flat_norm: Optional[torch.Tensor] = None
    if ref_datacube is not None:
        logger.debug("Hybrid: mean-subtracting reference library")
        ref_processed = ref_datacube.mean_subtract_()
        ref_data = torch.nan_to_num(ref_processed.tensor, nan=0.0)
        ref_flat = ref_data.view(n_ref, -1)  # (n_ref, ny*nx)
        ref_flat_norm = _normalize_library(ref_flat)

    # Pre-compute normalized target frames (for basis construction only).
    # The projection step always uses the original target_flat.
    target_flat_norm = _normalize_library(target_flat)

    if ref_flat is not None:
        logger.debug(
            f"Hybrid: n_ref={n_ref}, n_target={nk}, "
            f"hybrid_weight={hybrid_weight:.3g}  "
            "(per-frame L2 normalization applied to both libraries before SVD)"
        )

    # Select basis strategy
    try:
        basis_fn = _basis_strategies[method]
    except KeyError:
        error_msg = (
            f"Unknown method: {method!r}; "
            f"must be one of {list(_basis_strategies)}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Allocate output tensors on the same device as the science data
    residuals = torch.zeros(len(K_list), nk, ny, nx, device=data.device)
    if return_diagnostics:
        psf_estimates = torch.zeros(len(K_list), nk, ny, nx, device=data.device)
    # Accumulates per-frame projection coefficients (zero-padded to K_max)
    coefficients_out = torch.zeros(nk, K_max, device=data.device)

    # Group consecutive frames by PA for basis reuse
    pa_groups = _group_frames_by_pa(angles, pa_reuse_threshold)
    logger.info(
        f"Hybrid: {len(pa_groups)} PA-groups over {nk} frames "
        f"(pa_reuse_threshold={pa_reuse_threshold} deg, "
        f"annular_exclusion={annular_exclusion} deg)"
    )

    # Preserved from the last batch for KLIPDiagnostics
    Z_KL_last: Optional[torch.Tensor] = None

    for grp_idx, group in enumerate(pa_groups):
        # Representative PA: use the first frame of the batch
        rep_pa: float = angles[group[0]].item()
        g = len(group)

        # Boolean mask of valid target frames for this batch's library
        pa_diff = torch.abs(angles - rep_pa)              # (nk,)
        valid_mask: torch.Tensor = pa_diff > annular_exclusion  # (nk,) bool
        n_valid: int = int(valid_mask.sum().item())

        # Assemble the combined library for SVD using normalized frames.
        # Normalization ensures each frame contributes equal power to the KL
        # basis regardless of raw flux differences between the two datasets.
        # hybrid_weight upscales the target subset so it is not washed out
        # when n_ref >> n_valid.
        lib_parts_norm: List[torch.Tensor] = []
        if ref_flat_norm is not None:
            lib_parts_norm.append(ref_flat_norm)           # unit-norm ref rows
        if n_valid > 0:
            # Scale target frames by hybrid_weight for basis construction
            lib_parts_norm.append(
                target_flat_norm[valid_mask] * hybrid_weight
            )

        if not lib_parts_norm:
            # Degenerate edge case: annular_exclusion so wide that the entire
            # target falls inside the exclusion zone with no reference library.
            # Fall back to normalized full target to avoid a crash.
            logger.warning(
                f"Hybrid group {grp_idx}: empty library for "
                f"PA_rep={rep_pa:.2f} deg, "
                f"annular_exclusion={annular_exclusion} deg. "
                "Falling back to normalized full target library."
            )
            lib = target_flat_norm * hybrid_weight
        else:
            lib = torch.cat(lib_parts_norm, dim=0)         # (n_lib, ny*nx)

        logger.debug(
            f"Hybrid group {grp_idx}: basis lib n_rows={lib.shape[0]} "
            f"(ref={n_ref if ref_flat_norm is not None else 0}, "
            f"target_valid={n_valid}, hybrid_weight={hybrid_weight:.3g})"
        )

        # SVD rank cannot exceed the number of library frames
        k_max_eff = min(K_max, lib.shape[0])
        if k_max_eff < K_max:
            logger.debug(
                f"Hybrid group {grp_idx}: clamping K_max {K_max} → {k_max_eff} "
                f"(library size={lib.shape[0]}, PA_rep={rep_pa:.2f} deg)"
            )

        # Compute the KL basis for this PA batch
        logger.debug(
            f"Hybrid group {grp_idx}/{len(pa_groups)}: "
            f"frames={group}, n_lib={lib.shape[0]}, K_max_eff={k_max_eff}"
        )
        Z_KL = basis_fn(lib, k_max_eff)  # (ny*nx, k_max_eff)
        Z_KL_last = Z_KL

        # Vectorised projection over all frames in this batch
        group_frames = target_flat[group]  # (g, ny*nx)

        for list_idx, k in enumerate(K_list):
            k_eff = min(k, k_max_eff)
            V = Z_KL[:, :k_eff]                           # (ny*nx, k_eff)
            proj = group_frames @ V                        # (g, k_eff)
            ihat = proj @ V.T                              # (g, ny*nx)
            res = group_frames - ihat                      # (g, ny*nx)

            residuals[list_idx, group] = res.view(g, ny, nx)
            if return_diagnostics:
                psf_estimates[list_idx, group] = ihat.view(g, ny, nx)

            # Store projection coefficients for the largest K value
            # (zero-padded to K_max when k_max_eff < K_max)
            if k == K_list[-1]:
                coefficients_out[group, :k_eff] = proj

    # Restore NaN positions (NaNs survive mean_subtract_() in-place)
    logger.debug("Hybrid: restoring NaN mask")
    nan_mask = torch.isnan(datacube.tensor).unsqueeze(0)  # (1, nk, ny, nx)
    residuals = residuals.masked_fill(nan_mask, float("nan"))

    logger.info("Finished compute_residuals_hybrid")

    if return_diagnostics:
        psf_estimates = psf_estimates.masked_fill(nan_mask, float("nan"))

        # Build kl_basis from the last PA batch; zero-pad if clamped below K_max
        if Z_KL_last is not None:
            k_diag = Z_KL_last.shape[1]
            kl_basis = Z_KL_last.T.view(k_diag, ny, nx)
            if k_diag < K_max:
                pad = torch.zeros(K_max - k_diag, ny, nx, device=data.device)
                kl_basis = torch.cat([kl_basis, pad], dim=0)
        else:
            kl_basis = torch.zeros(K_max, ny, nx, device=data.device)

        diag = KLIPDiagnostics(
            kl_basis=kl_basis,               # (K_max, ny, nx)
            mean_frame=mean_frame,            # (ny, nx)
            psf_estimates=psf_estimates,      # (n_K, nk, ny, nx)
            residual_frames=residuals,        # (n_K, nk, ny, nx)
            coefficients=coefficients_out,    # (nk, K_max)
            K_list=K_list,
        )
        return residuals, diag

    return residuals


# ---------------------------------------------------------------------------
# Rotation / combination utilities
# ---------------------------------------------------------------------------

def batch_rotate(images: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rotates a stack of images by given angles via a single batched affine transform.

    This function performs efficient rotation of multiple images by batching
    the operation into a single GPU operation where possible.

    Args:
        images: Tensor of shape (n_k_values, nk, H, W).
        angles: 1D Tensor of length nk, angles in degrees for each frame.

    Returns:
        Rotated images, same shape as input.
    """
    n_k_values, N, H, W = images.shape

    # Create a flat batch of all images from all K values
    # Shape: (n_k_values*N, 1, H, W)
    flat_images = images.reshape(-1, H, W).unsqueeze(1)

    # Repeat angles for each K value
    repeated_angles = angles.repeat(n_k_values)
    angles_rad = torch.deg2rad(-repeated_angles)  # Negative for derotation

    # Create affine transformation matrices for all images
    cos = torch.cos(angles_rad)
    sin = torch.sin(angles_rad)

    # Create rotation matrices
    theta = torch.zeros((n_k_values*N, 2, 3), device=images.device)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos

    # Create sampling grid for all images
    grid = torch.nn.functional.affine_grid(
        theta, size=(n_k_values*N, 1, H, W), align_corners=False)

    # Apply grid sample to all images at once
    rotated_flat = torch.nn.functional.grid_sample(
        flat_images, grid, align_corners=False)

    # Reshape back to original format
    return rotated_flat.squeeze(1).reshape(n_k_values, N, H, W)


def _derotate_slow(residuals: torch.Tensor,
                   angles: torch.Tensor) -> torch.Tensor:
    """Performs slow per-frame derotation fallback using torchvision.rotate.

    This function is a fallback for when batch_rotate cannot be used, rotating
    each frame individually.

    Args:
        residuals: Tensor of shape (n_k_values, nk, H, W).
        angles: 1D Tensor of length nk.

    Returns:
        Derotated tensor of same shape.
    """
    from torchvision.transforms.functional import rotate, InterpolationMode

    n_k_values, nk, H, W = residuals.shape
    out = torch.zeros_like(residuals)
    for k_idx in range(n_k_values):
        for i in range(nk):
            frame = residuals[k_idx, i]
            out[k_idx, i] = rotate(
                frame.unsqueeze(0),
                float(-angles[i].item()),
                interpolation=InterpolationMode.BILINEAR
            ).squeeze(0)
    return out


def derotate_cube(residuals: torch.Tensor,
                  angles: Union[List[float], torch.Tensor],
                  batch: bool = True) -> torch.Tensor:
    """
    Derotate residual cube by negative angles.

    This function applies derotation to align the frames for subsequent combination.
    It dispatches to either batch processing or per-frame processing based on the
    batch parameter.

    Args:
        residuals: Tensor of shape (n_k_values, nk, H, W).
        angles: Sequence or 1D Tensor of length nk.
        batch: If True, use fast batched rotate; else fallback to per-frame loop.

    Returns:
        Tensor of same shape as `residuals`, but derotated.
    """
    return batch_rotate(residuals, angles) if batch else _derotate_slow(residuals, angles)


def combine_cube(derotated: torch.Tensor,
                 statistic: str = "mean") -> torch.Tensor:
    """Combine derotated frames along the temporal axis.

    This function applies a statistical operation (mean or median) to combine
    the derotated frames into a final result.

    Args:
        derotated: Tensor of shape (n_k_values, nk, H, W).
        statistic: 'mean' or 'median'.

    Returns:
        Tensor of shape (n_k_values, H, W).

    Raises:
        ValueError: If `statistic` is not 'mean' or 'median'.
    """
    # derotated: (n_k_values, nk, H, W)
    if statistic == "mean":
        return torch.nanmean(derotated, dim=1)
    elif statistic == "median":
        return torch.nanmedian(derotated, dim=1).values
    else:
        raise ValueError("statistic must be 'mean' or 'median'")


# ---------------------------------------------------------------------------
# RDI helpers: mask sharing and reference-frame selection
# ---------------------------------------------------------------------------

def _share_nan_mask(sci: DataTensor, ref: DataTensor) -> None:
    """Mask in ``ref`` every pixel that is NaN in any science frame.

    Operates **in place on** ``ref`` only; ``sci`` is never modified, so ADI
    results are unaffected.  After this call the reference library is zero
    (after ``nan_to_num``) exactly where the science frames are zero, which
    keeps the KL basis and the science projection on the same pixel set.

    Args:
        sci: Science DataTensor (already circularly masked, if applicable).
        ref: Reference DataTensor with the same ``(ny, nx)``.

    Raises:
        ValueError: If the image sizes differ.
    """
    if tuple(sci.shape[1:]) != tuple(ref.shape[1:]):
        raise ValueError(
            f"ref image size {tuple(ref.shape[1:])} != "
            f"science image size {tuple(sci.shape[1:])}")
    sci_nan = torch.isnan(sci.tensor).any(dim=0)          # (ny, nx)
    ref.tensor[:, sci_nan] = float("nan")


def select_reference_frames(sci: DataTensor,
                            ref: DataTensor,
                            n_keep: int) -> DataTensor:
    """Keep the ``n_keep`` reference frames most correlated with the science.

    Correlation (Pearson) is computed between each reference frame and the
    median science frame over the pixels that are valid in the science cube.

    Args:
        sci: Science DataTensor.
        ref: Reference DataTensor (same ``(ny, nx)``).
        n_keep: Number of reference frames to keep (clipped to ``ref.nk``).

    Returns:
        A new DataTensor holding the selected reference frames, ordered from
        most to least correlated.

    Raises:
        ValueError: If ``n_keep < 1`` or the image sizes differ.
    """
    if n_keep < 1:
        raise ValueError("n_keep must be >= 1")
    if tuple(sci.shape[1:]) != tuple(ref.shape[1:]):
        raise ValueError(
            f"ref image size {tuple(ref.shape[1:])} != "
            f"science image size {tuple(sci.shape[1:])}")

    n_keep = min(n_keep, ref.nk)
    sci_med = torch.nanmedian(sci.tensor, dim=0).values.flatten()   # (P,)
    valid = ~torch.isnan(sci.tensor).any(dim=0).flatten() & ~torch.isnan(sci_med)

    s = sci_med[valid]
    r = torch.nan_to_num(ref.tensor.flatten(1)[:, valid], nan=0.0)  # (N, P_valid)
    s = s - s.mean()
    r = r - r.mean(dim=1, keepdim=True)
    corr = (r @ s) / (r.norm(dim=1) * s.norm() + 1e-12)

    idx = torch.argsort(corr, descending=True)[:n_keep]
    logger.info(
        f"Selected {n_keep}/{ref.nk} reference frames "
        f"(corr range {corr[idx].min().item():.3f} .. {corr[idx].max().item():.3f})")
    return DataTensor(ref.tensor[idx], center=ref.center,
                      IWA=ref.IWA, OWA=ref.OWA)


class TorchKLIP:
    """
    High-level wrapper to run PSF subtraction and derotation.

    Attributes:
        device: torch.device for computation.
        datacube: Input DataTensor on `device`.
        angles: 1D Tensor of derotation angles.
        collect_metrics: If True, record timing metrics.
        store_intermediates: If True, keep intermediate arrays.
        metrics: Dict recording timing breakdown.
        IWA: Inner Working Angle for annular mask (optional).
        OWA: Outer Working Angle for annular mask (optional).
    """

    def __init__(self,
                 datacube: DataTensor,
                 angles: Union[List[float], torch.Tensor],
                 IWA: Optional[float] = None,
                 OWA: Optional[float] = None,
                 device: str = "cpu",
                 collect_metrics: bool = True,
                 store_intermediates: bool = False,
                 ref_data: Optional[DataTensor] = None) -> None:
        """Initializes the TorchKLIP processor.

        Args:
            datacube: DataTensor containing the science (target) frames.
            angles: List or tensor of derotation angles in degrees.
            IWA: Inner Working Angle for annular mask in pixels (optional).
                Takes precedence over ``datacube.IWA`` when provided.
            OWA: Outer Working Angle for annular mask in pixels (optional).
                Takes precedence over ``datacube.OWA`` when provided.
            device: Device to run computation on ('cpu' or 'cuda').
            collect_metrics: If True, record timing metrics.
            store_intermediates: If True, keep intermediate arrays.
            ref_data: Optional reference-star DataTensor for RDI or hybrid
                mode.  When supplied, ``klip_and_derotate`` with
                ``mode='rdi'`` or ``mode='hybrid'`` can use this library to
                augment the KL basis.

        Raises:
            ValueError: If number of angles doesn't match number of frames.
        """

        self.device = torch.device(device)
        self.datacube = datacube.to(device)
        self.ref_datacube: Optional[DataTensor] = (
            ref_data.to(device) if ref_data is not None else None
        )
        if isinstance(angles, torch.Tensor):
            self.angles = angles.clone().detach().to(
                dtype=torch.float32, device=self.device)
        else:
            if isinstance(angles, np.ndarray):
                angles = np.asarray(
                    angles, dtype=np.float32).newbyteorder('=').reshape(-1)
            self.angles = torch.tensor(
                angles, dtype=torch.float32, device=self.device).reshape(-1)
        self.collect_metrics = collect_metrics
        self.perfmon = PerformanceMonitor(self.device, enabled=collect_metrics)
        self.store_intermediates = store_intermediates
        self.metrics = {}

        # Prefer constructor arguments; fall back to datacube metadata
        self.IWA = IWA if IWA is not None else datacube.IWA
        self.OWA = OWA if OWA is not None else datacube.OWA

        if len(angles) != self.datacube.nk:
            logger.error(
                f"Angle mismatch: expected {self.datacube.nk}, got {len(angles)}")
            raise ValueError(
                f"Expected {self.datacube.nk} angles, got {len(angles)}.")
        if self.IWA is not None or self.OWA is not None:
            logger.info(
                f"Applying circular mask with IWA={self.IWA}, OWA={self.OWA}")
            self.datacube.mask_circular_(self.IWA, self.OWA)

        # RDI / hybrid: put the reference library on the same pixel set as the
        # (now masked) science cube.  Without this the KL basis is dominated by
        # pixels (e.g. inside the IWA) that the science frames have zeroed out.
        if self.ref_datacube is not None:
            _share_nan_mask(self.datacube, self.ref_datacube)
            logger.info(
                f"Reference library: {self.ref_datacube.nk} frames; "
                "science NaN mask applied")

    def _log_metrics(self):
        m = self.metrics
        logger.info("=== Performance Metrics ===")
        # memory
        for key in ("system_total_memory", "cpu_rss", "allocated_gpu_memory"):
            if key in m:
                logger.info(f"{key}: {bytes_to_human(m[key])}")
        # time
        if "total_time" in m:
            logger.info(f"Total time: {format_time(m['total_time'])}")
        for stage in ("psfsub_time", "derotate_time", "combine_time"):
            if stage in m:
                pct = m[stage] / m["total_time"] * 100
                logger.info(f"{stage}: {format_time(m[stage])} ({pct:.1f}%)")

    def klip_and_derotate(self,
                          K_klip: Union[int, List[int]],
                          method: str = "svd",
                          statistic: str = "mean",
                          isbatch: bool = True,
                          mode: str = "adi",
                          return_diagnostics: bool = False,
                          annular_exclusion: float = 0.0,
                          pa_reuse_threshold: float = 0.5,
                          hybrid_weight: float = 1.0,
                          normalize_ref: bool = False,
                          ) -> Union[torch.Tensor,
                                     Tuple[torch.Tensor, KLIPDiagnostics]]:
        """
        Perform KLIP PSF subtraction, derotation, and combination.

        This method runs the full KLIP workflow: PSF subtraction using KL modes,
        derotation of the residual frames, and statistical combination of the
        derotated frames.

        Args:
            K_klip: Number of KL modes to use (integer or list of integers).
            method: Method for computing basis ("svd", "pca", or "eigh").
            statistic: Statistic for combining frames ("mean" or "median").
            isbatch: Whether to use batch rotation.
            mode: Processing mode:

                * ``"adi"`` — self-reference (classic ADI).
                * ``"rdi"`` — external reference library supplied via
                  ``ref_data`` at construction time.
                * ``"hybrid"`` — ADI + RDI combined. For each PA batch the KL
                  basis is built from the union of the reference library and
                  the subset of target frames satisfying the angular exclusion
                  criterion ``|PA_target - PA_batch_rep| > annular_exclusion``.

            return_diagnostics: If True, return a ``(result, KLIPDiagnostics)``
                tuple instead of just ``result``.  The :class:`KLIPDiagnostics`
                object contains the following image-shaped arrays:

                * ``kl_basis`` – KL-mode eigenimages ``(K_max, ny, nx)``
                * ``mean_frame`` – temporal mean before subtraction ``(ny, nx)``
                * ``psf_estimates`` – reconstructed PSF per K ``(n_K, nk, ny, nx)``
                * ``residual_frames`` – pre-collapse residuals per K ``(n_K, nk, ny, nx)``
                * ``coefficients`` – projection weights ``(nk, K_max)``
                * ``K_list`` – sorted list of K values used

                The diagnostics are also stored as ``self.diagnostics`` for
                convenient post-hoc access.

                .. note::
                    In ``hybrid`` mode ``kl_basis`` and ``coefficients``
                    reflect the **last** PA batch.  All other diagnostic
                    arrays cover every frame.

            annular_exclusion: Angular exclusion threshold in degrees used only
                in ``hybrid`` mode.  Target frames within this PA distance of
                the batch representative are excluded from the self-reference
                subset of the library.  Default ``0.0`` (exclude only exact
                PA matches).
            pa_reuse_threshold: PA change in degrees that triggers a new SVD
                computation in ``hybrid`` mode.  Frames within the same PA
                batch share a single KL basis.  Default ``0.5`` deg.
            hybrid_weight: Relative amplitude of target frames versus
                reference frames when building the KL basis in ``hybrid``
                mode.  Both libraries are per-frame L2-normalized before
                this scale is applied, so ``1.0`` (default) gives equal
                per-frame weight.  Increase above ``1.0`` to prevent a
                large reference library from dominating the SVD.
                Has no effect in ``adi`` or ``rdi`` mode.
            normalize_ref: ``rdi`` mode only.  L2-normalise each reference
                frame before building the KL basis.  Default ``False``.
                (``hybrid`` mode always normalises.)

        Returns:
            * Default: 2D tensor ``(ny, nx)`` for single K, or 3D tensor
              ``(len(K_klip), ny, nx)`` for multiple K values.
            * ``return_diagnostics=True``: ``(result, KLIPDiagnostics)`` where
              *result* has the same shape as above.

        Raises:
            NotImplementedError: If *mode* is not ``'adi'``, ``'rdi'``, or
                ``'hybrid'``.
            ValueError: If ``mode='rdi'`` and no ``ref_data`` was provided at
                construction time.
        """
        # Validate processing mode
        mode = mode.lower()
        if mode not in ("adi", "rdi", "hybrid"):
            raise NotImplementedError(
                f"Mode '{mode}' not supported; choose 'adi', 'rdi', or 'hybrid'.")
        if mode == "rdi" and self.ref_datacube is None:
            raise ValueError(
                "mode='rdi' requires ref_data to be provided at construction time.")

        # Track if input is a single K value for final output shape
        is_single_k = isinstance(K_klip, int)
        k_list = [K_klip] if is_single_k else K_klip

        logger.info(
            f"Running KLIP with K={k_list}, method={method}, "
            f"statistic={statistic}, mode={mode}"
        )

        # For rdi/hybrid: if the reference has NaN pixels that are valid in the
        # science cube, mask those pixels in a working copy of the science so
        # both sides use exactly the same pixel set.  (ADI is untouched.)
        sci_dc = self.datacube
        if mode in ("rdi", "hybrid") and self.ref_datacube is not None:
            extra = (torch.isnan(self.ref_datacube.tensor).any(dim=0)
                     & ~torch.isnan(self.datacube.tensor).any(dim=0))
            if extra.any():
                logger.warning(
                    f"{int(extra.sum())} pixels are NaN in the reference but "
                    "valid in the science cube; masking them in the science "
                    "working copy.")
                sci_dc = DataTensor(self.datacube.tensor.clone(),
                                    center=self.datacube.center,
                                    IWA=self.datacube.IWA,
                                    OWA=self.datacube.OWA,
                                    copy=False)
                sci_dc.tensor[:, extra] = float("nan")

        # 1. PSF subtraction ------------------------------------------------
        self.perfmon.start_timer("total")
        self.perfmon.start_timer("psfsub")

        if mode == "hybrid":
            if self.store_intermediates:
                logger.warning(
                    "store_intermediates=True is not supported in hybrid mode; "
                    "intermediate arrays will not be stored."
                )
            if return_diagnostics:
                residuals, diagnostics = compute_residuals_hybrid(
                    sci_dc, k_list, self.angles, method,
                    annular_exclusion=annular_exclusion,
                    pa_reuse_threshold=pa_reuse_threshold,
                    return_diagnostics=True,
                    ref_datacube=self.ref_datacube,
                    hybrid_weight=hybrid_weight,
                )
                self.diagnostics = diagnostics
            else:
                residuals = compute_residuals_hybrid(
                    sci_dc, k_list, self.angles, method,
                    annular_exclusion=annular_exclusion,
                    pa_reuse_threshold=pa_reuse_threshold,
                    ref_datacube=self.ref_datacube,
                    hybrid_weight=hybrid_weight,
                )
        else:
            # ADI or RDI path (unchanged)
            ref_dc = self.ref_datacube if mode == "rdi" else None
            if return_diagnostics:
                residuals, diagnostics = compute_residuals(
                    sci_dc, k_list, method,
                    return_diagnostics=True, ref_datacube=ref_dc,
                    normalize_ref=normalize_ref,
                )
                self.diagnostics = diagnostics
            elif self.store_intermediates:
                out = compute_residuals(
                    sci_dc, k_list, method,
                    store_intermediates=True, ref_datacube=ref_dc,
                    normalize_ref=normalize_ref,
                )
                if isinstance(out, ResidualsWithIntermediates):
                    residuals = out.residuals
                    self.Z_KL = out.Z_KL
                    self.proj = out.proj
                    self.ihat = out.ihat
                    self.residual_flat = out.residual_flat
                else:
                    residuals, self.Z_KL, self.proj, self.ihat, self.residual_flat = out
                self.residuals = residuals
            else:
                residuals = compute_residuals(
                    sci_dc, k_list, method, ref_datacube=ref_dc,
                    normalize_ref=normalize_ref,
                )

        self.perfmon.stop_timer("psfsub")

        # 2. Derotation
        self.perfmon.start_timer("derotate")
        derot = derotate_cube(residuals, self.angles, isbatch)
        self.perfmon.stop_timer("derotate")

        # 3. Combination
        self.perfmon.start_timer("combine")
        result = combine_cube(derot, statistic)
        self.perfmon.stop_timer("combine")
        self.perfmon.stop_timer("total")

        if self.collect_metrics:
            self.perfmon.collect_memory_metrics()
            self.metrics = self.perfmon.get_metrics()

        #  Reapply final circular mask if IWA/OWA are set
        if self.IWA is not None or self.OWA is not None:
            mask2d = self.datacube.create_circular_mask(self.IWA, self.OWA)
            if result.ndim == 2:
                result = result.masked_fill(mask2d, float("nan"))
            elif result.ndim == 3:
                mask3d = mask2d.unsqueeze(0).expand_as(result)
                result = result.masked_fill(mask3d, float("nan"))

        # Only squeeze for single K value at the end
        if is_single_k:
            result = result.squeeze(0)

        if return_diagnostics:
            return result, self.diagnostics
        return result


__all__ = ["TorchKLIP", "KLIPDiagnostics", "compute_residuals_hybrid",
           "select_reference_frames"]

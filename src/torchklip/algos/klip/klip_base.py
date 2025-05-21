# File: src/torchklip/algos/klip/klip_base.py
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
                      store_intermediates: bool = False
                      ) -> Union[torch.Tensor, ResidualsWithIntermediates]:
    """
    Compute PSF-subtracted residual cube for given KL modes.

    This function implements the core KLIP (Karhunen-Loeve Image Processing) algorithm
    to perform PSF subtraction. It computes the KL basis, projects the input frames,
    and returns the residual frames after subtracting the reconstructions.

    Args:
        datacube: DataTensor containing the input frames.
        K_klip: Integer or list of integers specifying KL mode counts.
        method: Basis computation method, one of 'svd', 'pca', or 'eigh'.
        store_intermediates: If True, return intermediates in a ResidualsWithIntermediates.

    Returns:
        If store_intermediates is False, returns a Tensor of shape
        (n_k_values, nk, ny, nx) containing residuals.
        If store_intermediates is True, returns a ResidualsWithIntermediates
        with attributes (residuals, Z_KL, proj, ihat, residual_flat).

    Raises:
        ValueError: If K_klip list is empty, contains non-positive integers,
            or requests more modes than frames available.
        TypeError: If any entry in K_klip is not an integer.
        ValueError: If method is not in the supported strategies.
    """
    logger.info("Starting compute_residuals")
    nk, ny, nx = datacube.nk, datacube.ny, datacube.nx

    # Normalize and validate K_klip
    K_list = _normalize_and_validate_K(K_klip, nk, logger)
    K_max = K_list[-1]

    # Prepare data: mean‐subtract and replace NaNs
    logger.debug("Mean-subtracting datacube and replacing NaNs")
    processed = datacube.mean_subtract_()
    data = torch.nan_to_num(processed.tensor, nan=0.0)
    ref_flat = data.view(nk, -1)  # shape: (nk, ny*nx)

    # Compute KL basis
    logger.info(f"Computing KL basis with method={method!r}, K_max={K_max}")
    try:
        basis_fn = _basis_strategies[method]
    except KeyError:
        error_msg = f"Unknown method: {method!r}; must be one of {list(_basis_strategies)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    Z_KL = basis_fn(ref_flat, K_max)  # shape: (ny*nx, K_max)
    logger.debug(f"Basis shape: {tuple(Z_KL.shape)}")

    # Always compute a 4D tensor for residuals (n_k_values, nk, ny, nx)
    residuals = torch.zeros(len(K_list), nk, ny, nx, device=data.device)

    # Compute residuals for each K value
    for i, k in enumerate(K_list):
        logger.debug(f"Projecting and reconstructing for k={k}")

        # Compute projection and residuals
        proj_k = ref_flat @ Z_KL[:, :k]      # shape: (nk, k)
        ihat_k = proj_k @ Z_KL[:, :k].T      # shape: (nk, ny*nx)
        residual_flat_k = ref_flat - ihat_k  # shape: (nk, ny*nx)
        residuals[i] = residual_flat_k.view(nk, ny, nx)

        # Store intermediates for max K if requested
        if store_intermediates and k == K_max:
            proj_max = proj_k
            ihat_max = ihat_k
            residual_flat_max = residual_flat_k

    # Restore NaNs from original
    # shape: (1, nk, ny, nx) broadcast to (n_k_values, nk, ny, nx)
    logger.debug("Restoring NaNs to residuals")
    nan_mask = torch.isnan(datacube.tensor).unsqueeze(0)
    residuals = residuals.masked_fill(nan_mask, float("nan"))

    logger.info("Finished compute_residuals")
    # Return intermediates results if requested
    if store_intermediates:
        return ResidualsWithIntermediates(
            residuals, Z_KL, proj_max, ihat_max, residual_flat_max
        )
    return residuals


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
                 #   IWA: Optional[float] = None,
                 #   OWA: Optional[float] = None,
                 device: str = "cpu",
                 collect_metrics: bool = True,
                 store_intermediates: bool = False) -> None:
        """Initializes the TorchKLIP processor.

        Args:
            datacube: DataTensor containing the input frames.
            angles: List or tensor of derotation angles in degrees.
            IWA: Inner Working Angle for annular mask in pixels (optional).
            OWA: Outer Working Angle for annular mask in pixels (optional).
            device: Device to run computation on ('cpu' or 'cuda').
            collect_metrics: If True, record timing metrics.
            store_intermediates: If True, keep intermediate arrays.

        Raises:
            ValueError: If number of angles doesn't match number of frames.
        """

        self.device = torch.device(device)
        self.datacube = datacube.to(device)
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
        self.IWA = datacube.IWA
        self.OWA = datacube.OWA

        if len(angles) != self.datacube.nk:
            logger.error(
                f"Angle mismatch: expected {self.datacube.nk}, got {len(angles)}")
            raise ValueError(
                f"Expected {self.datacube.nk} angles, got {len(angles)}.")
        if self.IWA is not None or self.OWA is not None:
            logger.info(
                f"Applying circular mask with IWA={self.IWA}, OWA={self.OWA}")
            self.datacube.mask_circular_(self.IWA, self.OWA)

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
                          mode: str = "adi") -> torch.Tensor:
        """
        Perform KLIP PSF subtraction, derotation, and combination.

        This method runs the full KLIP workflow: PSF subtraction using KL modes,
        derotation of the residual frames, and statistical combination of the
        derotated frames.

        Args:
            K_klip: Number of KL modes to use (integer or list of integers)
            method: Method for computing basis ("svd", "pca", or "eigh")
            statistic: Statistic for combining frames ("mean" or "median")
            isbatch: Whether to use batch rotation
            mode: Processing mode ("adi" only supported currently)

        Returns:
            For single K_klip (integer): A 2D tensor of shape (ny, nx)
            For multiple K_klip (list): A 3D tensor of shape (len(K_klip), ny, nx)
        """
        # Validate processing mode
        if mode.lower() != "adi":
            raise NotImplementedError(
                f"Mode '{mode}' not supported; only 'adi' is implemented.")

        # Track if input is a single K value for final output shape
        is_single_k = isinstance(K_klip, int)
        k_list = [K_klip] if is_single_k else K_klip

        logger.info(
            f"Running KLIP with K={k_list}, method={method}, statistic={statistic}")

        # 1. PSF subtraction
        self.perfmon.start_timer("total")
        self.perfmon.start_timer("psfsub")
        if self.store_intermediates:
            out = compute_residuals(
                self.datacube, k_list, method, store_intermediates=True
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
            residuals = compute_residuals(self.datacube, k_list, method)
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
            # self.perfmon.print_metrics()
            # self._log_metrics()

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
            return result.squeeze(0)
        return result


__all__ = ["TorchKLIP"]

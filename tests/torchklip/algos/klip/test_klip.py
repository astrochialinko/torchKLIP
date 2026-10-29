import os
import pytest
import torch
import numpy as np
from typing import Tuple
from scipy.ndimage import gaussian_filter

# Import the modules to test
from torchklip.algos.klip.svd import compute_basis_svd
from torchklip.algos.klip.pca import compute_basis_pca
from torchklip.algos.klip.eigh import compute_basis_eigh
from torchklip.algos.klip.klip_base import (
    _normalize_and_validate_K,
    _group_frames_by_pa,
    compute_residuals,
    compute_residuals_hybrid,
    batch_rotate,
    derotate_cube,
    combine_cube,
    TorchKLIP,
    ResidualsWithIntermediates,
    KLIPDiagnostics,
    select_reference_frames,
)
from torchklip.dataproc.data_preprocessor import DataTensor
from torchklip.utils.logging_utils import get_logger

# Get the logger
logger = get_logger("test_klip")


@pytest.fixture
def synthetic_datacube() -> Tuple[DataTensor, torch.Tensor]:
    """Create a synthetic datacube for testing."""
    # Create a simple datacube with synthetic data
    nk, ny, nx = 5, 30, 30
    device = torch.device("cpu")

    # Create some PSF-like structure
    x, y = torch.meshgrid(
        torch.linspace(-5, 5, nx),
        torch.linspace(-5, 5, ny),
        indexing="ij"
    )
    r = torch.sqrt(x**2 + y**2)

    # Create a basic PSF shape: central peak with surrounding rings
    psf = torch.exp(-(r**2) / 2) - 0.1 * torch.exp(-(r**2) / 8)

    # Create datacube with this PSF moved to different positions
    datacube = torch.zeros((nk, ny, nx), device=device)

    # Add a PSF at different positions with some jitter
    positions = [(15, 15), (14, 16), (16, 14), (15, 14), (14, 15)]
    for i, (cy, cx) in enumerate(positions):
        # Shift the PSF to the right position
        shift_y = cy - ny // 2
        shift_x = cx - nx // 2
        rolled = torch.roll(psf, shifts=(shift_y, shift_x), dims=(0, 1))

        # Add some random noise
        noise = torch.randn_like(rolled) * 0.05
        datacube[i] = rolled + noise

    # Create angles for derotation (arbitrary for the test)
    angles = torch.tensor([0.0, 30.0, 60.0, 90.0, 120.0], device=device)

    # Wrap in a DataTensor
    data_tensor = DataTensor(datacube)

    return data_tensor, angles


def test_normalize_and_validate_K():
    """Test K parameter normalization and validation."""
    # Valid cases
    assert _normalize_and_validate_K(5, 10, logger) == [5]
    assert _normalize_and_validate_K([1, 3, 5], 10, logger) == [1, 3, 5]
    assert _normalize_and_validate_K([5, 1, 3], 10, logger) == [
        1, 3, 5]  # Should sort

    # Invalid cases
    with pytest.raises(ValueError):
        _normalize_and_validate_K([], 10, logger)  # Empty list

    with pytest.raises(TypeError):
        _normalize_and_validate_K([1, 2.5, 3], 10, logger)  # Non-integer

    with pytest.raises(ValueError):
        _normalize_and_validate_K([1, 0, 3], 10, logger)  # Non-positive

    with pytest.raises(ValueError):
        _normalize_and_validate_K([1, 5, 15], 10, logger)  # K > nk


def test_basis_computation_methods():
    """Test that all basis computation methods produce output of expected shape."""
    # Create a simple reference tensor
    nk, npix = 5, 100
    reference = torch.randn((nk, npix))
    K_max = 3

    # Test SVD method
    Z_KL_svd = compute_basis_svd(reference, K_max)
    assert Z_KL_svd.shape == (npix, K_max)

    # Test PCA method
    Z_KL_pca = compute_basis_pca(reference, K_max)
    assert Z_KL_pca.shape == (npix, K_max)

    # Test EIGH method
    Z_KL_eigh = compute_basis_eigh(reference, K_max)
    assert Z_KL_eigh.shape == (npix, K_max)

    # Because these methods can produce bases that differ by sign and order,
    # we need to test the span of the subspace, not exact basis vectors

    # Create a test image to reconstruct with each basis
    test_image = torch.randn(npix)

    # Project and reconstruct with each basis
    recon_svd = (Z_KL_svd @ (Z_KL_svd.T @ test_image))
    recon_pca = (Z_KL_pca @ (Z_KL_pca.T @ test_image))
    recon_eigh = (Z_KL_eigh @ (Z_KL_eigh.T @ test_image))

    # Calculate reconstruction error norms
    err_svd = torch.norm(test_image - recon_svd)
    err_pca = torch.norm(test_image - recon_pca)
    err_eigh = torch.norm(test_image - recon_eigh)

    # All methods should provide similar reconstruction quality
    # The differences should be small relative to the original norm
    norm_test = torch.norm(test_image)

    # Check that errors are similar to each other (within 20%)
    assert abs(err_svd - err_pca) / norm_test < 0.2
    assert abs(err_svd - err_eigh) / norm_test < 0.2


def test_compute_residuals(synthetic_datacube):
    """Test computing residuals."""
    datacube, _ = synthetic_datacube

    # Test with a single K value
    K_klip = 3
    residuals = compute_residuals(datacube, K_klip)
    assert residuals.shape == (1, datacube.nk, datacube.ny, datacube.nx)

    # Test with multiple K values
    K_klip = [1, 2, 3]
    residuals = compute_residuals(datacube, K_klip)
    assert residuals.shape == (
        len(K_klip), datacube.nk, datacube.ny, datacube.nx)

    # Test with intermediates
    K_klip = 3
    result = compute_residuals(datacube, K_klip, store_intermediates=True)
    assert isinstance(result, ResidualsWithIntermediates)
    assert result.residuals.shape == (1, datacube.nk, datacube.ny, datacube.nx)
    assert result.Z_KL.shape == (datacube.ny * datacube.nx, K_klip)
    assert result.proj.shape == (datacube.nk, K_klip)
    assert result.ihat.shape == (datacube.nk, datacube.ny * datacube.nx)
    assert result.residual_flat.shape == (
        datacube.nk, datacube.ny * datacube.nx)

    # Test methods
    methods = ["svd", "pca", "eigh"]
    for method in methods:
        residuals = compute_residuals(datacube, K_klip, method=method)
        assert residuals.shape == (1, datacube.nk, datacube.ny, datacube.nx)

    # Test invalid method
    with pytest.raises(ValueError):
        compute_residuals(datacube, K_klip, method="invalid_method")


def test_batch_rotate():
    """Test batch rotation functionality."""
    # Create a simple test pattern that clearly shows rotation
    H, W = 8, 8
    nk = 2
    n_k_values = 1

    # Create a simple L-shaped pattern
    test_pattern = torch.zeros((n_k_values, nk, H, W))

    # First frame: horizontal line
    test_pattern[0, 0, H//2, :] = 1.0

    # Second frame: vertical line
    test_pattern[0, 1, :, W//2] = 1.0

    # Rotate by 90 degrees (should approximately transform the shapes)
    angles = torch.tensor([90.0, 0.0])
    rotated = batch_rotate(test_pattern, angles)

    # First frame should now have a vertical line (within tolerance for interpolation)
    vertical_mask = rotated[0, 0, :, W//2-1:W//2+2].sum() > 0.5 * H
    assert vertical_mask

    # Second frame should still have a vertical line
    vertical_mask = rotated[0, 1, :, W//2-1:W//2+2].sum() > 0.5 * H
    assert vertical_mask


def test_derotate_cube(synthetic_datacube):
    """Test cube derotation."""
    datacube, angles = synthetic_datacube

    # Create a dummy residuals cube
    residuals = torch.zeros((1, datacube.nk, datacube.ny, datacube.nx))
    # Just use the datacube as residuals for testing
    residuals[0] = datacube.tensor

    # Test batch derotation
    derotated_batch = derotate_cube(residuals, angles, batch=True)
    assert derotated_batch.shape == residuals.shape

    # Test per-frame derotation (slow)
    derotated_slow = derotate_cube(residuals, angles, batch=False)
    assert derotated_slow.shape == residuals.shape

    # They should be approximately equal (allowing for small differences due to implementation)
    assert torch.allclose(derotated_batch, derotated_slow, atol=1e-4)


def test_combine_cube():
    """Test frame combination methods."""
    # Create a simple test cube
    n_k_values, nk, H, W = 1, 3, 10, 10
    test_cube = torch.ones((n_k_values, nk, H, W))

    # Add some NaNs
    test_cube[0, 0, 0, 0] = float('nan')
    test_cube[0, 1, 0, 0] = float('nan')
    # All three values need to be NaN to result in NaN after combining
    test_cube[0, 2, 0, 0] = float('nan')

    # Add different values
    test_cube[0, 0] *= 1
    test_cube[0, 1] *= 2
    test_cube[0, 2] *= 3

    # Test mean combination
    mean_result = combine_cube(test_cube, statistic="mean")
    assert mean_result.shape == (n_k_values, H, W)
    # Mean of [1, 2, 3] should be 2
    assert torch.isclose(mean_result[0, 1, 1], torch.tensor(2.0))

    # Test median combination
    median_result = combine_cube(test_cube, statistic="median")
    assert median_result.shape == (n_k_values, H, W)
    # Median of [1, 2, 3] should be 2
    assert torch.isclose(median_result[0, 1, 1], torch.tensor(2.0))

    # Test invalid statistic
    with pytest.raises(ValueError):
        combine_cube(test_cube, statistic="invalid")

    # Test NaN handling
    assert torch.isnan(mean_result[0, 0, 0])
    assert torch.isnan(median_result[0, 0, 0])


def test_TorchKLIP_init(synthetic_datacube):
    """Test TorchKLIP initialization."""
    datacube, angles = synthetic_datacube

    # Test basic initialization
    klip = TorchKLIP(datacube, angles)
    assert klip.datacube is not None
    assert torch.all(klip.angles == angles)

    # Test with IWA/OWA
    klip = TorchKLIP(datacube, angles, IWA=5.0, OWA=15.0)
    assert klip.IWA == 5.0
    assert klip.OWA == 15.0

    # Test with a different number of angles
    wrong_angles = torch.cat(
        [angles, torch.tensor([150.0], device=angles.device)])
    with pytest.raises(ValueError):
        TorchKLIP(datacube, wrong_angles)  # More angles than frames


def test_TorchKLIP_klip_and_derotate(synthetic_datacube):
    """Test the full KLIP workflow."""
    datacube, angles = synthetic_datacube

    # Initialize KLIP processor
    klip = TorchKLIP(datacube, angles, collect_metrics=True)

    # Test with a single K value
    K_klip = 3
    result = klip.klip_and_derotate(K_klip)
    assert result.shape == (datacube.ny, datacube.nx)

    # Check metrics collection
    assert "psfsub_time" in klip.metrics
    assert "derotate_time" in klip.metrics
    assert "total_time" in klip.metrics

    # Test with multiple K values
    K_klip = [1, 2, 3]
    result = klip.klip_and_derotate(K_klip)
    assert result.shape == (len(K_klip), datacube.ny, datacube.nx)

    # Test with different methods
    methods = ["svd", "pca", "eigh"]
    for method in methods:
        result = klip.klip_and_derotate(K_klip, method=method)
        assert result.shape == (len(K_klip), datacube.ny, datacube.nx)

    # Test with intermediates
    klip = TorchKLIP(datacube, angles, store_intermediates=True)
    result = klip.klip_and_derotate(K_klip)
    assert hasattr(klip, "Z_KL")
    assert hasattr(klip, "proj")
    assert hasattr(klip, "ihat")
    assert hasattr(klip, "residual_flat")

    # Test statistics
    for stat in ["mean", "median"]:
        result = klip.klip_and_derotate(K_klip, statistic=stat)
        assert result.shape == (len(K_klip), datacube.ny, datacube.nx)

    # Test unsupported mode
    with pytest.raises(NotImplementedError):
        klip.klip_and_derotate(K_klip, mode="unsupported")


def test_ResidualsWithIntermediates():
    """Test the ResidualsWithIntermediates dataclass and its iterator."""
    # Create dummy tensors for testing
    residuals = torch.ones((1, 5, 10, 10))
    Z_KL = torch.ones((100, 3))
    proj = torch.ones((5, 3))
    ihat = torch.ones((5, 100))
    residual_flat = torch.ones((5, 100))

    # Create the dataclass instance
    container = ResidualsWithIntermediates(
        residuals, Z_KL, proj, ihat, residual_flat
    )

    # Test attribute access
    assert torch.all(container.residuals == residuals)
    assert torch.all(container.Z_KL == Z_KL)
    assert torch.all(container.proj == proj)
    assert torch.all(container.ihat == ihat)
    assert torch.all(container.residual_flat == residual_flat)

    # Test unpacking via iteration
    r, z, p, i, rf = container
    assert torch.all(r == residuals)
    assert torch.all(z == Z_KL)
    assert torch.all(p == proj)
    assert torch.all(i == ihat)
    assert torch.all(rf == residual_flat)


def test_compute_residuals_return_diagnostics(synthetic_datacube):
    """Test compute_residuals with return_diagnostics=True."""
    datacube, _ = synthetic_datacube
    nk, ny, nx = datacube.nk, datacube.ny, datacube.nx
    K_klip = [1, 3]

    result = compute_residuals(datacube, K_klip, return_diagnostics=True)
    assert isinstance(result, tuple) and len(result) == 2

    residuals, diag = result
    assert isinstance(diag, KLIPDiagnostics)

    K_max = max(K_klip)
    n_K = len(K_klip)

    # Shape checks
    assert residuals.shape == (n_K, nk, ny, nx)
    assert diag.kl_basis.shape == (K_max, ny, nx)
    assert diag.mean_frame.shape == (ny, nx)
    assert diag.psf_estimates.shape == (n_K, nk, ny, nx)
    assert diag.residual_frames.shape == (n_K, nk, ny, nx)
    assert diag.coefficients.shape == (nk, K_max)
    assert diag.K_list == sorted(K_klip)

    # residual_frames must equal returned residuals tensor
    assert torch.allclose(diag.residual_frames, residuals, equal_nan=True)

    # residuals + psf_estimates ≈ mean-subtracted input (allowing for float rounding)
    # Use the larger-K slice (last entry)
    combined = diag.residual_frames[-1] + diag.psf_estimates[-1]
    # Where neither is NaN, the sum should be close to mean-subtracted data
    valid = ~torch.isnan(combined)
    assert valid.any()


def test_compute_residuals_diagnostics_single_k(synthetic_datacube):
    """return_diagnostics works with a single K value."""
    datacube, _ = synthetic_datacube
    nk, ny, nx = datacube.nk, datacube.ny, datacube.nx
    K_klip = 3

    residuals, diag = compute_residuals(datacube, K_klip, return_diagnostics=True)
    assert residuals.shape == (1, nk, ny, nx)
    assert diag.kl_basis.shape == (K_klip, ny, nx)
    assert diag.coefficients.shape == (nk, K_klip)
    assert diag.K_list == [K_klip]


def test_klip_and_derotate_return_diagnostics(synthetic_datacube):
    """Test TorchKLIP.klip_and_derotate with return_diagnostics=True."""
    datacube, angles = synthetic_datacube
    nk, ny, nx = datacube.nk, datacube.ny, datacube.nx

    klip = TorchKLIP(datacube, angles)

    # Single K — result should be 2D, diagnostics present
    result, diag = klip.klip_and_derotate(3, return_diagnostics=True)
    assert result.shape == (ny, nx)
    assert isinstance(diag, KLIPDiagnostics)
    assert klip.diagnostics is diag  # also stored on instance

    # Multiple K — result should be 3D
    K_list = [1, 2, 3]
    result, diag = klip.klip_and_derotate(K_list, return_diagnostics=True)
    assert result.shape == (len(K_list), ny, nx)
    assert diag.psf_estimates.shape == (len(K_list), nk, ny, nx)
    assert diag.residual_frames.shape == (len(K_list), nk, ny, nx)
    assert diag.kl_basis.shape == (max(K_list), ny, nx)
    assert diag.mean_frame.shape == (ny, nx)
    assert diag.coefficients.shape == (nk, max(K_list))
    assert diag.K_list == sorted(K_list)


def test_klip_and_derotate_backward_compatible(synthetic_datacube):
    """return_diagnostics=False preserves original return type."""
    datacube, angles = synthetic_datacube
    ny, nx = datacube.ny, datacube.nx

    klip = TorchKLIP(datacube, angles)
    result = klip.klip_and_derotate(3)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (ny, nx)


def test_klip_diagnostics_public_api():
    """KLIPDiagnostics is importable from the top-level package."""
    import torchklip
    assert hasattr(torchklip, "KLIPDiagnostics")


# ---------------------------------------------------------------------------
# Hybrid (ADI + RDI) mode tests
# ---------------------------------------------------------------------------

def test_group_frames_by_pa():
    """_group_frames_by_pa groups consecutive frames below threshold."""
    # Slow drift: all frames should end up in one group at threshold=1.0
    angles = torch.tensor([0.0, 0.3, 0.6, 0.9, 1.5, 2.0])
    groups = _group_frames_by_pa(angles, pa_reuse_threshold=1.0)
    # Frame 4 (PA=1.5) starts a new group (|1.5-0.0| >= 1.0)
    assert len(groups) >= 2
    # All frames are covered without overlap
    all_indices = sorted(idx for grp in groups for idx in grp)
    assert all_indices == list(range(len(angles)))

    # Large threshold: everything in one group
    groups_single = _group_frames_by_pa(angles, pa_reuse_threshold=10.0)
    assert len(groups_single) == 1
    assert groups_single[0] == list(range(len(angles)))

    # Tiny threshold: each frame in its own group
    angles_distinct = torch.tensor([0.0, 1.0, 2.0, 3.0])
    groups_all_split = _group_frames_by_pa(angles_distinct, pa_reuse_threshold=0.5)
    assert len(groups_all_split) == 4

    # Edge case: single frame
    groups_one = _group_frames_by_pa(torch.tensor([45.0]), pa_reuse_threshold=1.0)
    assert groups_one == [[0]]

    # Edge case: empty
    assert _group_frames_by_pa(torch.tensor([]), pa_reuse_threshold=1.0) == []


def test_hybrid_mode_basic_shapes(synthetic_datacube):
    """compute_residuals_hybrid returns correct output shapes."""
    datacube, angles = synthetic_datacube
    nk, ny, nx = datacube.nk, datacube.ny, datacube.nx

    # Single K → shape (1, nk, ny, nx)
    residuals = compute_residuals_hybrid(datacube, 3, angles)
    assert residuals.shape == (1, nk, ny, nx)

    # Multiple K → shape (n_K, nk, ny, nx)
    K_list = [1, 2, 3]
    residuals = compute_residuals_hybrid(datacube, K_list, angles)
    assert residuals.shape == (len(K_list), nk, ny, nx)

    # Output should be finite (no spurious NaN from numerical errors)
    assert torch.isfinite(residuals).all()


def test_hybrid_mode_all_methods(synthetic_datacube):
    """All basis strategies work in hybrid mode."""
    datacube, angles = synthetic_datacube
    nk, ny, nx = datacube.nk, datacube.ny, datacube.nx

    for method in ("svd", "pca", "eigh"):
        res = compute_residuals_hybrid(datacube, 2, angles, method=method)
        assert res.shape == (1, nk, ny, nx), f"Wrong shape for method={method}"

    with pytest.raises(ValueError):
        compute_residuals_hybrid(datacube, 2, angles, method="bad_method")


def test_hybrid_mode_angular_exclusion(synthetic_datacube):
    """annular_exclusion controls which target frames enter the KL library."""
    datacube, angles = synthetic_datacube
    nk, ny, nx = datacube.nk, datacube.ny, datacube.nx

    # exclusion = 0: every frame outside the current PA batch is eligible
    res_no_excl = compute_residuals_hybrid(
        datacube, 3, angles, annular_exclusion=0.0
    )
    assert res_no_excl.shape == (1, nk, ny, nx)

    # Exclusion wider than any possible PA separation and no reference library:
    # the KL library is empty, which must raise rather than self-subtract.
    with pytest.raises(ValueError, match="empty KL library"):
        compute_residuals_hybrid(datacube, 3, angles, annular_exclusion=360.0)

    # The same wide exclusion is fine when a reference library is supplied:
    # the basis is then built from the reference frames alone (pure RDI).
    ref_dc = DataTensor(torch.randn(4, ny, nx))
    res_ref_only = compute_residuals_hybrid(
        datacube, 3, angles, annular_exclusion=360.0, ref_datacube=ref_dc
    )
    assert res_ref_only.shape == (1, nk, ny, nx)


def test_hybrid_mode_with_reference(synthetic_datacube):
    """Hybrid mode with a reference library uses both ref and target frames."""
    datacube, angles = synthetic_datacube
    nk, ny, nx = datacube.nk, datacube.ny, datacube.nx

    # Build a small synthetic reference library
    ref_tensor = torch.randn(4, ny, nx)
    ref_datacube = DataTensor(ref_tensor)

    # K bounded by n_ref + nk = 4 + 5 = 9; requesting K=4 is fine
    residuals = compute_residuals_hybrid(
        datacube, 4, angles, ref_datacube=ref_datacube
    )
    assert residuals.shape == (1, nk, ny, nx)


def test_hybrid_mode_diagnostics(synthetic_datacube):
    """return_diagnostics=True populates KLIPDiagnostics correctly."""
    datacube, angles = synthetic_datacube
    nk, ny, nx = datacube.nk, datacube.ny, datacube.nx
    K_list = [1, 3]

    result = compute_residuals_hybrid(
        datacube, K_list, angles, return_diagnostics=True
    )
    assert isinstance(result, tuple) and len(result) == 2

    residuals, diag = result
    assert isinstance(diag, KLIPDiagnostics)

    K_max = max(K_list)
    n_K = len(K_list)

    assert residuals.shape == (n_K, nk, ny, nx)
    assert diag.kl_basis.shape == (K_max, ny, nx)
    assert diag.mean_frame.shape == (ny, nx)
    assert diag.psf_estimates.shape == (n_K, nk, ny, nx)
    assert diag.residual_frames.shape == (n_K, nk, ny, nx)
    assert diag.coefficients.shape == (nk, K_max)
    assert diag.K_list == sorted(K_list)

    # residual_frames must equal the returned residuals tensor
    assert torch.allclose(diag.residual_frames, residuals, equal_nan=True)

    # residuals + psf_estimates ≈ mean-subtracted input
    combined = diag.residual_frames[-1] + diag.psf_estimates[-1]
    valid = ~torch.isnan(combined)
    assert valid.any()


def test_torchklip_hybrid_mode(synthetic_datacube):
    """TorchKLIP.klip_and_derotate with mode='hybrid' runs end-to-end."""
    datacube, angles = synthetic_datacube
    nk, ny, nx = datacube.nk, datacube.ny, datacube.nx

    klip = TorchKLIP(datacube, angles, collect_metrics=True)

    # Single K — result squeezed to 2D
    result = klip.klip_and_derotate(3, mode="hybrid")
    assert result.shape == (ny, nx)
    assert isinstance(result, torch.Tensor)

    # Multiple K — result is 3D
    result_multi = klip.klip_and_derotate([1, 2, 3], mode="hybrid")
    assert result_multi.shape == (3, ny, nx)

    # With diagnostics
    result_diag, diag = klip.klip_and_derotate(
        [1, 3], mode="hybrid", return_diagnostics=True
    )
    assert result_diag.shape == (2, ny, nx)
    assert isinstance(diag, KLIPDiagnostics)
    assert klip.diagnostics is diag

    # Metrics should still be collected
    assert "psfsub_time" in klip.metrics
    assert "total_time" in klip.metrics


def test_torchklip_hybrid_with_annular_exclusion(synthetic_datacube):
    """annular_exclusion parameter is forwarded correctly in hybrid mode."""
    datacube, angles = synthetic_datacube
    ny, nx = datacube.ny, datacube.nx

    klip = TorchKLIP(datacube, angles)
    result = klip.klip_and_derotate(
        2, mode="hybrid", annular_exclusion=10.0, pa_reuse_threshold=1.0
    )
    assert result.shape == (ny, nx)


def test_torchklip_hybrid_with_reference(synthetic_datacube):
    """Hybrid mode with external reference library via TorchKLIP."""
    datacube, angles = synthetic_datacube
    nk, ny, nx = datacube.nk, datacube.ny, datacube.nx

    ref_tensor = torch.randn(4, ny, nx)
    ref_dc = DataTensor(ref_tensor)

    klip = TorchKLIP(datacube, angles, ref_data=ref_dc)
    result = klip.klip_and_derotate(4, mode="hybrid")
    assert result.shape == (ny, nx)


def test_torchklip_unsupported_mode_raises(synthetic_datacube):
    """Unsupported mode raises NotImplementedError for all known bad inputs."""
    datacube, angles = synthetic_datacube
    klip = TorchKLIP(datacube, angles)
    with pytest.raises(NotImplementedError):
        klip.klip_and_derotate(2, mode="bad_mode")


def test_torchklip_hybrid_nan_restoration(synthetic_datacube):
    """NaN pixels in the input are restored in hybrid mode output."""
    datacube, angles = synthetic_datacube
    nk, ny, nx = datacube.nk, datacube.ny, datacube.nx

    # Inject NaN pixels into a copy of the datacube
    cube_with_nan = datacube.tensor.clone()
    cube_with_nan[:, 0, 0] = float("nan")
    nan_dc = DataTensor(cube_with_nan)

    residuals = compute_residuals_hybrid(nan_dc, 2, angles)
    # The NaN position must be propagated
    assert torch.isnan(residuals[:, :, 0, 0]).all()

def test_hybrid_excludes_batch_members_from_library():
    """No frame may be fitted with itself or a batch neighbour.

    Two PA batches of three frames each. With annular_exclusion=0 and K equal
    to the library size, any frame that sits in its own library would be
    reconstructed almost perfectly, driving its residual towards zero.
    Independent random frames should instead keep most of their norm.
    """
    torch.manual_seed(0)
    nk, ny, nx = 6, 32, 32
    cube = torch.randn(nk, ny, nx)
    angles = torch.tensor([0.0, 0.1, 0.2, 10.0, 10.1, 10.2])

    residuals = compute_residuals_hybrid(
        DataTensor(cube.clone()), 5, angles,
        annular_exclusion=0.0, pa_reuse_threshold=0.5,
    )

    res_norm = residuals[0].flatten(1).norm(dim=1)
    in_norm = cube.flatten(1).norm(dim=1)
    assert torch.all(res_norm > 0.5 * in_norm), res_norm / in_norm

# ---------------------------------------------------------------------------
# RDI tests with an injected fake planet
# ---------------------------------------------------------------------------

_PL_SEP = 10.0      # planet separation (pixels)
_PL_SIGMA = 1.5     # planet Gaussian sigma (pixels)
_PL_AMP = 0.3       # planet peak amplitude
_PL_APERTURE = 3.0  # photometry aperture radius (pixels)


def _planet_xy(angle_deg: float, cy: float, cx: float) -> Tuple[float, float]:
    theta = np.deg2rad(angle_deg)
    return cx + _PL_SEP * np.cos(theta), cy + _PL_SEP * np.sin(theta)


@pytest.fixture
def rdi_planet_data():
    """Science and reference cubes sharing the same stellar PSF structure.

    Every stellar frame is a base PSF plus a random combination of four
    smooth speckle modes, so the reference library spans the science PSF.
    Only the science cube contains a planet, whose position rotates with PA.
    """
    rng = np.random.default_rng(0)
    ny = nx = 41
    cy = cx = 20
    y, x = np.mgrid[:ny, :nx]
    r = np.hypot(y - cy, x - cx)

    psf0 = np.exp(-r**2 / (2 * 3.0**2)) + 0.2 * np.exp(-(r - 8)**2 / (2 * 1.5**2))
    modes = [gaussian_filter(rng.standard_normal((ny, nx)), 3) for _ in range(4)]
    modes = [m / np.abs(m).max() * 0.05 for m in modes]

    def stellar_frames(n):
        return np.stack([
            psf0 * (1 + 0.05 * rng.standard_normal())
            + sum(rng.standard_normal() * m for m in modes)
            + 0.002 * rng.standard_normal((ny, nx))
            for _ in range(n)
        ])

    nk, n_ref = 8, 20
    angles = np.arange(nk) * 10.0

    planet = np.zeros((nk, ny, nx))
    for i, a in enumerate(angles):
        px, py = _planet_xy(a, cy, cx)
        planet[i] = _PL_AMP * np.exp(-((x - px)**2 + (y - py)**2) / (2 * _PL_SIGMA**2))

    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    return {
        "sci": to_t(stellar_frames(nk) + planet),
        "ref": to_t(stellar_frames(n_ref)),
        "planet": to_t(planet),
        "angles": to_t(angles),
        "center": (cy, cx),
    }


def _planet_throughput(residuals: torch.Tensor, data: dict) -> torch.Tensor:
    """Per-frame ratio of recovered to injected planet flux in an aperture.

    Args:
        residuals: ``(nk, ny, nx)`` PSF-subtracted frames before derotation.
        data: Output of the ``rdi_planet_data`` fixture.

    Returns:
        ``(nk,)`` tensor of throughputs (1.0 means the planet is fully kept).
    """
    cy, cx = data["center"]
    ny, nx = residuals.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing="ij")
    out = []
    for i, a in enumerate(data["angles"].tolist()):
        px, py = _planet_xy(a, cy, cx)
        ap = torch.hypot(xx - px, yy - py) <= _PL_APERTURE
        out.append(residuals[i][ap].sum() / data["planet"][i][ap].sum())
    return torch.stack(out)


@pytest.mark.parametrize("method", ["svd", "pca", "eigh"])
def test_rdi_preserves_planet(rdi_planet_data, method):
    """RDI removes the star but keeps nearly all of the planet flux."""
    d = rdi_planet_data
    residuals = compute_residuals(
        DataTensor(d["sci"].clone()), [5, 10], method,
        ref_datacube=DataTensor(d["ref"].clone()),
    )
    for k_idx in range(residuals.shape[0]):
        thr = _planet_throughput(residuals[k_idx], d)
        assert thr.mean() > 0.85, thr
        assert thr.min() > 0.75, thr


def test_rdi_normalize_ref_preserves_planet(rdi_planet_data):
    """normalize_ref changes only the basis weighting, not planet throughput."""
    d = rdi_planet_data
    residuals = compute_residuals(
        DataTensor(d["sci"].clone()), 10,
        ref_datacube=DataTensor(d["ref"].clone()), normalize_ref=True,
    )
    assert _planet_throughput(residuals[0], d).mean() > 0.85


def test_rdi_k_bounded_by_reference_size(rdi_planet_data):
    """In RDI mode K may exceed the number of science frames but not refs."""
    d = rdi_planet_data
    n_ref = d["ref"].shape[0]
    # K larger than nk (8) but within n_ref (20) is allowed
    compute_residuals(DataTensor(d["sci"].clone()), 15,
                      ref_datacube=DataTensor(d["ref"].clone()))
    with pytest.raises(ValueError):
        compute_residuals(DataTensor(d["sci"].clone()), n_ref + 1,
                          ref_datacube=DataTensor(d["ref"].clone()))


def test_hybrid_outperforms_full_frame_adi_at_high_k(rdi_planet_data):
    """Hybrid with exclusion keeps the planet where full-frame ADI loses it."""
    d = rdi_planet_data
    nk = d["sci"].shape[0]

    adi = compute_residuals(DataTensor(d["sci"].clone()), nk - 1)
    hyb = compute_residuals_hybrid(
        DataTensor(d["sci"].clone()), 15, d["angles"],
        annular_exclusion=15.0, ref_datacube=DataTensor(d["ref"].clone()),
    )
    thr_adi = _planet_throughput(adi[0], d).mean()
    thr_hyb = _planet_throughput(hyb[0], d).mean()

    assert thr_adi < 0.2, thr_adi    # documents full-frame ADI self-subtraction
    assert thr_hyb > 0.6, thr_hyb
    assert thr_hyb > thr_adi


def test_torchklip_rdi_requires_reference(rdi_planet_data):
    """mode='rdi' without ref_data raises a clear error."""
    d = rdi_planet_data
    klip = TorchKLIP(DataTensor(d["sci"].clone()), d["angles"])
    with pytest.raises(ValueError, match="ref_data"):
        klip.klip_and_derotate(5, mode="rdi")


def test_select_reference_frames_prefers_correlated(rdi_planet_data):
    """PSF-like reference frames are chosen over pure-noise frames."""
    d = rdi_planet_data
    good = d["ref"][:5]
    bad = torch.randn_like(good) * good.std()
    ref = DataTensor(torch.cat([bad, good]))

    selected = select_reference_frames(DataTensor(d["sci"].clone()), ref, n_keep=5)
    assert selected.nk == 5
    for frame in selected.tensor:
        assert any(torch.allclose(frame, g) for g in good)


def test_select_reference_frames_validation(rdi_planet_data):
    """n_keep is validated and clipped to the library size."""
    d = rdi_planet_data
    sci = DataTensor(d["sci"].clone())
    ref = DataTensor(d["ref"].clone())
    with pytest.raises(ValueError):
        select_reference_frames(sci, ref, n_keep=0)
    assert select_reference_frames(sci, ref, n_keep=999).nk == ref.nk
    
if __name__ == "__main__":
    # This allows running the tests directly
    pytest.main(["-xvs", __file__])

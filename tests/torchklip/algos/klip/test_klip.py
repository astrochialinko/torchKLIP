import os
import pytest
import torch
import numpy as np
from typing import Tuple

# Import the modules to test
from torchklip.algos.klip.svd import compute_basis_svd
from torchklip.algos.klip.pca import compute_basis_pca
from torchklip.algos.klip.eigh import compute_basis_eigh
from torchklip.algos.klip.klip_base import (
    _normalize_and_validate_K,
    compute_residuals,
    batch_rotate,
    derotate_cube,
    combine_cube,
    TorchKLIP,
    ResidualsWithIntermediates
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


if __name__ == "__main__":
    # This allows running the tests directly
    pytest.main(["-xvs", __file__])

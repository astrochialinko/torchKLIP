# tests/torchklip/dataproc/test_data_preprocessor.py
# Run with: pytest tests/torchklip/dataproc/test_data_preprocessor.py -v
import pytest
import torch
import numpy as np
from typing import Tuple
from unittest.mock import patch, Mock

from torchklip.dataproc.data_preprocessor import DataTensor


@pytest.fixture
def sample_data() -> Tuple[np.ndarray, torch.Tensor]:
    """Create sample data for testing."""
    np_data = np.random.randn(5, 10, 12).astype(np.float32)
    torch_data = torch.randn(5, 10, 12)
    return np_data, torch_data


@pytest.fixture
def data_tensor(sample_data) -> DataTensor:
    """Create a DataTensor instance for testing."""
    _, torch_data = sample_data
    return DataTensor(torch_data)


class TestDataTensor:
    """Test suite for the DataTensor class."""

    class TestInitialization:
        """Test DataTensor initialization and validation."""

        def test_init_with_torch_tensor(self, sample_data):
            """Test initialization with PyTorch tensor."""
            _, torch_data = sample_data
            dt = DataTensor(torch_data)
            assert torch.allclose(dt.tensor, torch_data)
            assert dt.center == ((torch_data.shape[2] - 1) // 2,
                                 (torch_data.shape[1] - 1) // 2)

        def test_init_with_numpy_array(self, sample_data):
            """Test initialization with NumPy array."""
            np_data, _ = sample_data
            dt = DataTensor(np_data)
            assert isinstance(dt.tensor, torch.Tensor)
            assert torch.allclose(dt.tensor, torch.from_numpy(np_data))

        def test_init_with_copy_true(self, sample_data):
            """Test that copy=True creates a copy of input data."""
            _, torch_data = sample_data
            dt = DataTensor(torch_data, copy=True)
            dt.tensor[0, 0, 0] = 999
            assert torch_data[0, 0, 0] != 999

        def test_init_with_copy_false(self, sample_data):
            """Test that copy=False shares the tensor data."""
            _, torch_data = sample_data
            dt = DataTensor(torch_data, copy=False)
            dt.tensor[0, 0, 0] = 999
            assert torch_data[0, 0, 0] == 999

        def test_init_with_custom_center(self, sample_data):
            """Test initialization with custom center coordinates."""
            _, torch_data = sample_data
            center = (3, 4)
            dt = DataTensor(torch_data, center=center)
            assert dt.center == center

        def test_init_with_numpy_center(self, sample_data):
            """Test initialization with NumPy array as center."""
            _, torch_data = sample_data
            center = np.array([3, 4])
            dt = DataTensor(torch_data, center=center)
            # The center should be converted to a tuple internally
            # Check if center is correctly converted to tuple
            assert dt.center == (3, 4) or dt.center == tuple(center)

        def test_invalid_dimensions(self):
            """Test error handling for invalid tensor dimensions."""
            with pytest.raises(ValueError, match="Expected 3D tensor"):
                DataTensor(torch.randn(10, 10))

        def test_invalid_data_type(self):
            """Test error handling for invalid input type."""
            with pytest.raises(TypeError, match="Expected np.ndarray or torch.Tensor"):
                DataTensor([1, 2, 3])

        def test_invalid_center_length(self, sample_data):
            """Test error handling for invalid center length."""
            _, torch_data = sample_data
            with pytest.raises(ValueError, match="center must have exactly 2 values"):
                DataTensor(torch_data, center=(1, 2, 3))

        def test_invalid_center_type(self, sample_data):
            """Test error handling for invalid center type."""
            _, torch_data = sample_data
            with pytest.raises(ValueError, match="Invalid center parameter"):
                DataTensor(torch_data, center="invalid")

        def test_center_out_of_bounds_warning(self, sample_data, caplog):
            """Test warning for center coordinates outside tensor bounds."""
            _, torch_data = sample_data
            with patch('torchklip.dataproc.data_preprocessor.logger.warning') as mock_warning:
                DataTensor(torch_data, center=(100, 100))
                mock_warning.assert_called_once()

    class TestProperties:
        """Test DataTensor properties."""

        def test_shape_properties(self, data_tensor):
            """Test shape-related properties."""
            assert data_tensor.shape == data_tensor.tensor.shape
            assert data_tensor.nk == data_tensor.tensor.shape[0]
            assert data_tensor.ny == data_tensor.tensor.shape[1]
            assert data_tensor.nx == data_tensor.tensor.shape[2]
            assert data_tensor.ndim == 3

        def test_center_properties(self, data_tensor):
            """Test center coordinate properties."""
            cx, cy = data_tensor.center
            assert data_tensor.cx == cx
            assert data_tensor.cy == cy

        def test_len(self, data_tensor):
            """Test __len__ method."""
            assert len(data_tensor) == data_tensor.nk

        def test_getitem(self, data_tensor):
            """Test __getitem__ method."""
            assert torch.allclose(data_tensor[0], data_tensor.tensor[0])
            assert torch.allclose(
                data_tensor[:, 1:3], data_tensor.tensor[:, 1:3])

    class TestMethods:
        """Test DataTensor methods."""

        def test_to_device(self, data_tensor):
            """Test moving tensor to device."""
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

            # Test non-inplace
            new_dt = data_tensor.to(device)
            assert new_dt.tensor.device.type == device
            assert data_tensor.tensor.device.type == 'cpu'

            # Test inplace
            original_device = data_tensor.tensor.device
            data_tensor.to(device, inplace=True)
            assert data_tensor.tensor.device.type == device

        def test_create_circular_mask(self, data_tensor):
            """Test circular mask creation."""
            # Test inner mask only
            mask_inner = data_tensor.create_circular_mask(IWA=3.0)
            assert mask_inner.shape == (data_tensor.ny, data_tensor.nx)

            # Test outer mask only
            mask_outer = data_tensor.create_circular_mask(OWA=5.0)
            assert mask_outer.shape == (data_tensor.ny, data_tensor.nx)

            # Test both inner and outer
            mask_both = data_tensor.create_circular_mask(IWA=3.0, OWA=5.0)
            assert mask_both.shape == (data_tensor.ny, data_tensor.nx)

            # Test caching
            mask_cached = data_tensor.create_circular_mask(IWA=3.0)
            assert torch.allclose(mask_inner, mask_cached)

        def test_mask_circular_(self, data_tensor):
            """Test in-place circular masking."""
            original_data = data_tensor.tensor.clone()

            # Apply mask
            result = data_tensor.mask_circular_(IWA=2.0, OWA=5.0)

            # Check that it returns self
            assert result is data_tensor

            # Check that values are masked
            mask = data_tensor.create_circular_mask(IWA=2.0, OWA=5.0)
            assert torch.isnan(data_tensor.tensor[mask.unsqueeze(
                0).expand_as(data_tensor.tensor)]).all()

            # Check that unmasked values are unchanged
            assert torch.allclose(
                data_tensor.tensor[~mask.unsqueeze(
                    0).expand_as(data_tensor.tensor)],
                original_data[~mask.unsqueeze(
                    0).expand_as(data_tensor.tensor)],
                equal_nan=True
            )

        def test_mask_circular_no_params_warning(self, data_tensor):
            """Test warning when mask_circular_ is called without parameters."""
            with patch('torchklip.dataproc.data_preprocessor.logger.warning') as mock_warning:
                data_tensor.mask_circular_()
                mock_warning.assert_called_once()

        def test_mean_subtract_(self, data_tensor):
            """Test in-place mean subtraction."""
            # Apply mean subtraction
            result = data_tensor.mean_subtract_()

            # Check that it returns self
            assert result is data_tensor

            # Check that each frame has zero mean
            for i in range(data_tensor.nk):
                assert torch.abs(torch.nanmean(data_tensor.tensor[i])) < 1e-6

        def test_mean_subtract_with_nans(self):
            """Test mean subtraction with NaN values."""
            data = torch.randn(3, 5, 5)
            data[0, 1, 1] = float('nan')
            data[1, 2, 2] = float('nan')

            dt = DataTensor(data)
            dt.mean_subtract_()

            # Check that mean is still approximately zero (ignoring NaNs)
            for i in range(dt.nk):
                assert torch.abs(torch.nanmean(dt.tensor[i])) < 1e-6

        def test_flatten(self, data_tensor):
            """Test flattening operation."""
            flattened = data_tensor.flatten()
            assert flattened.shape == (
                data_tensor.nk, data_tensor.ny * data_tensor.nx)

            flattened_start0 = data_tensor.flatten(start_dim=0)
            assert flattened_start0.shape == (
                data_tensor.nk * data_tensor.ny * data_tensor.nx,)

        def test_nan_to_num(self, data_tensor):
            """Test NaN replacement."""
            # Add NaN values
            data_tensor.tensor[0, 0, 0] = float('nan')
            data_tensor.tensor[1, 1, 1] = float('inf')
            data_tensor.tensor[2, 2, 2] = float('-inf')

            # Replace NaN values
            result = data_tensor.nan_to_num(nan=0.0)

            # Check that it returns a new DataTensor
            assert result is not data_tensor
            assert not torch.isnan(result.tensor).any()
            assert not torch.isinf(result.tensor).any()
            assert result.tensor[0, 0, 0] == 0.0

        def test_clear_cache(self, data_tensor):
            """Test cache clearing."""
            # Create some masks to populate cache
            data_tensor.create_circular_mask(IWA=3.0)
            data_tensor.create_circular_mask(OWA=5.0)

            assert len(data_tensor._mask_cache) == 2

            # Clear cache
            data_tensor.clear_cache()
            assert len(data_tensor._mask_cache) == 0

    class TestEdgeCases:
        """Test edge cases and special scenarios."""

        def test_numpy_array_non_native_endianness(self):
            """Test handling of non-native endianness."""
            # Create array with non-native byte order
            arr = np.array(np.random.randn(3, 4, 5), dtype='>f4')  # Big-endian
            dt = DataTensor(arr)
            assert dt.tensor.dtype == torch.float32

        def test_numpy_array_type_conversion(self):
            """Test automatic conversion to float32."""
            arr = np.random.randn(3, 4, 5).astype(np.float64)
            dt = DataTensor(arr)
            assert dt.tensor.dtype == torch.float32

        def test_empty_mask(self, data_tensor):
            """Test mask creation with no parameters."""
            # The current implementation has a bug where mask is not defined
            # when both IWA and OWA are None. This test documents this behavior.
            # A better implementation would return a mask with all False values
            # or raise a more descriptive error.

            with pytest.raises(UnboundLocalError):
                data_tensor.create_circular_mask()

            # Note: The expected behavior might be to return a mask with all False values
            # since no pixels should be masked when neither IWA nor OWA are specified

        def test_method_chaining(self, sample_data):
            """Test that methods can be chained."""
            _, torch_data = sample_data
            dt = DataTensor(torch_data)
            result = dt.mask_circular_(IWA=2.0).mean_subtract_()
            assert result is dt

        def test_mask_with_extreme_values(self, data_tensor):
            """Test mask with very large/small values."""
            # Very large OWA (should mask everything)
            mask_large = data_tensor.create_circular_mask(OWA=1000.0)
            assert not mask_large.any()  # No pixels should be masked

            # Very small IWA (should mask very little)
            mask_small = data_tensor.create_circular_mask(IWA=0.1)
            assert mask_small.sum() <= 1  # At most the center pixel

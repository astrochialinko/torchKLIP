# torchklip/dataproc/data_preprocessor.py
from dataclasses import dataclass, replace
from typing import Optional, Tuple, Union, List, Dict, Any

import torch
import numpy as np

# local modules
from ..utils.logging_utils import get_logger

# Get a logger specific to this module
logger = get_logger(__name__.split('.')[-1])


@dataclass
class DataTensor:
    """Container for multi-dimensional data with chainable transforms.

    Supports astronomical data processing with three dimensional metadata
    The class converts input data to PyTorch tensors internally while preserving
    the original dimensions and metadata.

    Attributes:
        tensor: The underlying PyTorch tensor storing the data.
        center: Optional tuple specifying the (x, y) center coordinates.
        IWA: Optional float for the inner working angle.
        OWA: Optional float for the outer working angle.
        copy: Boolean indicating whether to copy the input tensor.
    """
    tensor: Union[torch.Tensor, np.ndarray]
    center: Optional[Tuple[int, int]] = None
    IWA: Optional[float] = None
    OWA: Optional[float] = None
    copy: bool = True

    def __post_init__(self):

        # Validate and prepare the input tensor
        tensor = self._prepare_tensor(self.tensor)
        self._validate_dimensions(tensor)

        # Store the tensor and shape properties
        object.__setattr__(self, 'tensor', tensor)
        self._set_shape_properties(tensor)

        # Initialize center if not provided
        if self.center is None:
            self._set_default_center()
        else:
            self._validate_and_set_center()

        # if self.IWA is not None or self.OWA is not None:
        #     logger.info(
        #         f"Applying circular mask with IWA={self.IWA}, OWA={self.OWA}")
        #     self.tensor.mask_circular_()

        # internal cache for masks
        self._mask_cache: dict[Tuple[Optional[float],
                                     Optional[float]], torch.Tensor] = {}

    def _prepare_tensor(self, tensor: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Convert input to PyTorch tensor and handle copying."""
        # Only copy if user requested
        if self.copy:
            if isinstance(tensor, np.ndarray):
                tensor = tensor.copy()
            elif isinstance(tensor, torch.Tensor):
                tensor = tensor.clone()

        # Convert numpy array to PyTorch tensor if necessary
        if isinstance(tensor, np.ndarray):
            tensor = self._numpy_to_torch(tensor)

        elif not isinstance(tensor, torch.Tensor):
            error_msg = f"Expected np.ndarray or torch.Tensor, got {type(tensor).__name__}"
            logger.error(error_msg)
            raise TypeError(error_msg)

        return tensor

    def _numpy_to_torch(self, array: np.ndarray) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor with proper dtype and endianness."""
        if array.dtype != np.float32:
            array = array.astype(np.float32, copy=True)

        # Swap bytes if array is not in native endianness
        if array.dtype.byteorder not in ('=', '|'):
            array = array.byteswap().newbyteorder()

        return torch.from_numpy(array)

    def _validate_dimensions(self, tensor: torch.Tensor) -> None:
        """Validate that the tensor has exactly 3 dimensions."""
        if tensor.ndim != 3:
            error_msg = f"Expected 3D tensor, got shape {tensor.shape}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _set_shape_properties(self, tensor: torch.Tensor) -> None:
        """Set shape-related properties."""
        nk, ny, nx = tensor.shape
        object.__setattr__(self, '_nk', nk)
        object.__setattr__(self, '_ny', ny)
        object.__setattr__(self, '_nx', nx)

    def _set_default_center(self) -> None:
        """Set center to the geometric center of the tensor."""
        object.__setattr__(
            self, 'center', ((self._nx - 1) // 2, (self._ny - 1) // 2))

    def _validate_and_set_center(self) -> None:
        """Validate and set the provided center coordinates."""
        try:

            # Validate center type and length
            if isinstance(self.center, (tuple, list)):
                if len(self.center) != 2:
                    raise ValueError(
                        f"center must have exactly 2 values, got {len(self.center)}")
                center_tuple = tuple(int(coord) for coord in self.center)

            # Validate numpy array center
            elif isinstance(self.center, np.ndarray):
                if self.center.size != 2:
                    raise ValueError(
                        f"center array must have exactly 2 values, got {self.center.size}")
                center_tuple = tuple(int(coord)
                                     for coord in self.center.flatten())
            # Validate unsupported types
            else:
                raise TypeError(
                    f"center must be tuple, list, or numpy array, got {type(self.center).__name__}")

            # Validate bounds
            cx, cy = center_tuple
            if not (0 <= cx < self.nx and 0 <= cy < self.ny):
                logger.warning(
                    f"center {center_tuple} is outside tensor bounds {(self.nx, self.ny)}")

            object.__setattr__(self, 'center', center_tuple)

        except (ValueError, TypeError) as e:
            error_msg = f"Invalid center parameter: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the tensor."""
        return self.tensor.shape

    @property
    def nk(self) -> int:
        """Returns the size of the first dimension."""
        return self._nk

    @property
    def ny(self) -> int:
        """Returns the height of the tensor."""
        return self._ny

    @property
    def nx(self) -> int:
        """Returns the width of the tensor."""
        return self._nx

    @property
    def cx(self) -> int:
        """Returns the x-coordinate of the center."""
        return self.center[0]

    @property
    def cy(self) -> int:
        """Returns the y-coordinate of the center."""
        return self.center[1]

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions."""
        return self.tensor.ndim

    def __len__(self) -> int:
        """Returns the size of the first dimension."""
        return self.nk

    def __getitem__(self, idx: Any) -> torch.Tensor:
        """Supports various indexing operations.

        Args:
            idx: Index or slice to access the underlying tensor.

        Returns:
            The indexed tensor portion.
        """
        return self.tensor[idx]

    def to(self,
            device: Union[str, torch.device],
            non_blocking: bool = False,
            inplace: bool = False) -> Union["DataTensor", torch.Tensor]:
        """Moves the underlying tensor to the specified device.

        Args:
            device: The destination device.
            non_blocking: Whether to perform asynchronous transfer if possible.
            inplace: If True, modifies this instance instead of creating a copy.

        Returns:
            Either self (if inplace) or a new DataTensor on the target device.
        """
        if inplace:
            self.tensor = self.tensor.to(device, non_blocking=non_blocking)
            return self
        else:
            # shallow copy of metadata, deep copy of tensor pointer
            return DataTensor(self.tensor.to(device, non_blocking=non_blocking),
                              center=self.center)

    def clear_cache(self):
        """Clear the mask cache to free memory."""
        self._mask_cache.clear()

    def create_circular_mask(self,
                             IWA: Optional[float] = None,
                             OWA: Optional[float] = None) -> torch.Tensor:
        """Returns a 2D boolean mask for circular regions.

        Creates a mask where True indicates pixels to be masked (dist < IWA or 
        dist > OWA). Results are cached based on the (IWA, OWA) tuple.

        Args:
            IWA: Inner working angle (minimum radius).
            OWA: Outer working angle (maximum radius).

        Returns:
            A boolean tensor of shape (ny, nx) where True values indicate masked pixels.
        """
        IWA = self.IWA if IWA is None else IWA
        OWA = self.OWA if OWA is None else OWA
        key = (IWA, OWA)
        if key in self._mask_cache:
            return self._mask_cache[key]

        # Create coordinate grids
        y, x = torch.meshgrid(
            torch.arange(self.ny, device=self.tensor.device,
                         dtype=torch.float32),
            torch.arange(self.nx, device=self.tensor.device,
                         dtype=torch.float32),
            indexing='ij'
        )
        cx, cy = self.center
        dist2 = (x - cx)**2 + (y - cy)**2

        # Create mask based on IWA and OWA
        if IWA is not None:
            mask = dist2 <= IWA**2
            if OWA is not None:
                mask = mask | (dist2 >= OWA**2)
        elif OWA is not None:
            mask = dist2 >= OWA**2

        self._mask_cache[key] = mask
        return mask

    def mask_circular_(self,
                       IWA: Optional[float] = None,
                       OWA: Optional[float] = None) -> "DataTensor":
        """Applies a circular mask in-place.

        Sets values outside the specified annular region to NaN.

        Args:
            IWA: Inner working angle (minimum radius).
            OWA: Outer working angle (maximum radius).

        Returns:
            Self, for method chaining.
        """
        IWA = self.IWA if IWA is None else IWA
        OWA = self.OWA if OWA is None else OWA

        if IWA is None and OWA is None:
            logger.warning(
                "mask_circular_ called without IWA or OWA - no masking applied")
            return self

        # Create mask and expand to 3D
        mask2d = self.create_circular_mask(IWA, OWA)
        mask3d = mask2d.unsqueeze(0).expand_as(self.tensor)
        self.tensor.masked_fill_(mask3d, float('nan'))

        return self

    def mean_subtract_(self) -> "DataTensor":
        """Performs in-place, frame-wise mean subtraction.

        Subtracts the spatial mean from each frame, ignoring NaNs.
        After this operation, each frame has zero spatial mean.

        Returns:
            Self, for method chaining.
        """
        flat = self.tensor.view(self.nk, -1)
        means = torch.nanmean(flat, dim=1, keepdim=True)  # shape (nk, 1)

        # broadcast & subtract in-place
        flat.sub_(means)
        return self

    def flatten(self, start_dim: int = 1) -> torch.Tensor:
        """Flattens the tensor from start_dim onwards.

        Args:
            start_dim: First dimension to flatten (default: 1)

        Returns:
            Flattened torch.Tensor
        """
        return self.tensor.flatten(start_dim=start_dim)

    def nan_to_num(self, nan: float = 0.0, posinf: float = None, neginf: float = None) -> "DataTensor":
        """Replaces NaN, positive infinity, and negative infinity values.

        Args:
            nan: Value to replace NaNs with (default: 0.0)
            posinf: Value to replace positive infinity with (default: largest finite value)
            neginf: Value to replace negative infinity with (default: smallest finite value)

        Returns:
            A new DataTensor with replaced values
        """
        result = torch.nan_to_num(
            self.tensor, nan=nan, posinf=posinf, neginf=neginf)
        return replace(self, tensor=result)


__all__ = ["DataTensor"]

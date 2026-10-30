# torchklip/utils/data.py
"""General-purpose array manipulation utilities."""

import numpy as np

__all__ = ["median_bin"]


def median_bin(cube: np.ndarray, factor: int) -> np.ndarray:
    """Median-bin *cube* along axis-0 by *factor* frames.

    Args:
        cube   : ``(N, H, W)`` image cube.
        factor : Number of consecutive frames to collapse into one median frame.

    Returns:
        ``(ceil(N / factor), H, W)`` median-binned cube.
    """
    return np.array([
        np.median(cube[i: i + factor], axis=0)
        for i in range(0, len(cube), factor)
    ])

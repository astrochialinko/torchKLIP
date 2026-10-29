import numpy as np
import pytest

from torchklip.utils.data import median_bin


def test_median_bin_values():
    cube = np.arange(4, dtype=float).reshape(4, 1, 1) * np.ones((4, 2, 2))
    out = median_bin(cube, 2)
    assert out.shape == (2, 2, 2)
    np.testing.assert_allclose(out[:, 0, 0], [0.5, 2.5])


def test_median_bin_uneven_length():
    """The last bin may hold fewer frames: output length is ceil(N / factor)."""
    cube = np.random.default_rng(0).standard_normal((5, 3, 3))
    out = median_bin(cube, 2)
    assert out.shape == (3, 3, 3)
    np.testing.assert_allclose(out[-1], cube[-1])


def test_median_bin_factor_one_is_identity():
    cube = np.random.default_rng(1).standard_normal((4, 3, 3))
    np.testing.assert_allclose(median_bin(cube, 1), cube)
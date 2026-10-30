import numpy as np
from astropy.io import fits

from torchklip.io import save_fits_outputs


def test_save_fits_outputs(tmp_path):
    k_list = [1, 5]
    result = np.random.default_rng(0).standard_normal((2, 8, 10))
    kl_basis = np.random.default_rng(1).standard_normal((5, 8, 10))

    save_fits_outputs(result, kl_basis, k_list, tmp_path,
                      label="test", method="svd", mode="rdi")

    with fits.open(tmp_path / "test_residuals_cube.fits") as hdul:
        assert hdul[0].data.shape == (2, 8, 10)
        assert hdul[0].header["METHOD"] == "SVD"
        assert hdul[0].header["MODE"] == "RDI"
        assert hdul[0].header["KLIST"] == "1,5"
        np.testing.assert_allclose(hdul[0].data, result.astype(np.float32))

    for k_idx, k in enumerate(k_list):
        with fits.open(tmp_path / f"test_combined_K{k}.fits") as hdul:
            assert hdul[0].data.shape == (8, 10)
            assert hdul[0].header["K_KLIP"] == k

    with fits.open(tmp_path / "test_kl_basis.fits") as hdul:
        assert hdul[0].data.shape == (5, 8, 10)
        assert hdul[0].header["KMAX"] == 5

def test_save_fits_outputs_creates_dir_and_sets_bunit(tmp_path):
    out = tmp_path / "nested" / "dir"
    result = np.zeros((1, 4, 4))
    kl_basis = np.zeros((2, 4, 4))
    save_fits_outputs(result, kl_basis, [2], str(out), label="t",
                      method="svd", mode="adi", bunit="contrast")
    with fits.open(out / "t_residuals_cube.fits") as hdul:
        assert hdul[0].header["BUNIT"] == "contrast"
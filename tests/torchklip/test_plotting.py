import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch

from torchklip.plotting import (
    plot_method_comparison,
    plot_noise_histogram,
    plot_residual_multi_planet,
)

PLANETS = [
    {"label": "b", "x": 70.0, "y": 50.0, "color": "cyan"},
    {"label": "c", "x": 50.0, "y": 25.0, "color": "magenta"},
]


def test_plot_noise_histogram(tmp_path):
    path = tmp_path / "hist.png"
    plot_noise_histogram(torch.randn(500), "b", snr=5.0, signal=1.0,
                         noise_std=0.2, save_path=path)
    assert path.exists()


def test_plot_noise_histogram_empty_input_writes_nothing(tmp_path):
    path = tmp_path / "hist.png"
    plot_noise_histogram(torch.tensor([]), "b", 0.0, 0.0, 0.0, path)
    assert not path.exists()


def test_plot_residual_multi_planet_with_nan(tmp_path):
    """Runs on an image with NaN (e.g. a masked IWA) and returns one SNR per planet."""
    img = np.random.default_rng(0).standard_normal((101, 101))
    img[45:56, 45:56] = np.nan
    path = tmp_path / "residual.png"

    snr = plot_residual_multi_planet(
        img, k=5, planets=PLANETS, fwhm=4.0, center_coords=(50, 50),
        iwa=5.0, statistic="mean", save_path=path, label="test",
    )

    assert path.exists()
    assert set(snr) == {"b", "c"}
    for p in PLANETS:
        assert (tmp_path / f"test_K5_{p['label']}_noise_hist.png").exists()


def test_plot_method_comparison(tmp_path):
    k_list = [1, 5, 10]
    all_snr = {
        mode: {p["label"]: {k: float(k) for k in k_list} for p in PLANETS}
        for mode in ("adi", "rdi")
    }
    plot_method_comparison(all_snr, k_list, PLANETS, tmp_path, title_suffix="test")
    assert (tmp_path / "ADI_vs_RDI_SNR_curve.png").exists()
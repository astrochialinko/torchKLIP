# torchklip/plotting.py
"""Visualization utilities for KLIP PSF-subtraction results.

Functions
---------
plot_noise_histogram
    Histogram of noise-aperture pixel values with a Gaussian overlay.
plot_residual_multi_planet
    Residual image annotated with per-planet SNR, apertures, and noise histograms.
plot_method_comparison
    Multi-mode (ADI / RDI / Hybrid) SNR-vs-K comparison plot.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from .utils.image_plot import plot_image
from .utils.snr import compute_snr_components, draw_apertures

__all__ = [
    "plot_noise_histogram",
    "plot_residual_multi_planet",
    "plot_method_comparison",
]


def plot_noise_histogram(
    noise_pixels: torch.Tensor,
    planet_label: str,
    snr: float,
    signal: float,
    noise_std: float,
    save_path: Path,
) -> None:
    """Plot a histogram of noise-aperture pixel values with a Gaussian overlay.

    Args:
        noise_pixels  : 1-D tensor of raw pixel values from all valid noise
                        apertures (as returned by :func:`~torchklip.utils.snr.compute_snr_components`).
        planet_label  : Short label for the planet (used in the title).
        snr           : Pre-computed SNR value (for annotation).
        signal        : Planet aperture value (shown as a vertical line).
        noise_std     : Std of noise aperture reduced values (for annotation).
        save_path     : Output file path (PNG recommended).
    """
    from scipy.stats import norm as scipy_norm

    pixels_np = noise_pixels.cpu().numpy()
    if len(pixels_np) == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    _, bins, _ = ax.hist(
        pixels_np, bins=30, density=True, alpha=0.7,
        color="steelblue", edgecolor="white", label="Noise pixels",
    )

    # Gaussian MLE fit
    mu, sigma = scipy_norm.fit(pixels_np)
    x_fit = np.linspace(bins[0], bins[-1], 300)
    ax.plot(
        x_fit, scipy_norm.pdf(x_fit, mu, sigma),
        "r-", linewidth=2,
        label=f"Gaussian fit\n(μ={mu:.1e}, σ={sigma:.1e})",
    )

    ax.axvline(
        signal, color="limegreen", linestyle="--", linewidth=1.5,
        label=f"Signal = {signal:.1e}",
    )

    ax.set_xlabel("Pixel value")
    ax.set_ylabel("Density")
    ax.set_title(
        f"{planet_label}: noise-aperture pixel distribution\n"
        f"SNR = {snr:.2f}  |  Sig = {signal:.1e}  |  Noise σ = {noise_std:.1e}",
        fontsize=9,
    )
    ax.legend(fontsize=8, framealpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residual_multi_planet(
    combined: np.ndarray,
    k: int,
    planets: List[dict],
    fwhm: float,
    center_coords: Tuple[int, int],
    iwa: float,
    statistic: str,
    save_path: Path,
    label: str = "KLIP",
    draw_circles: bool = True,
    shared_colorbar: bool = True,
    exclude_nearest: int = 2,
    exclude_other_planets: bool = True,
) -> Dict[str, float]:
    """Plot combined residual image with per-planet SNR annotations.

    Also saves one noise-aperture histogram per planet alongside the main
    residual image (see :func:`plot_noise_histogram`).

    Args:
        combined             : ``(H, W)`` mean/median-combined residual frame.
        k                    : Number of KL modes (for plot title).
        planets              : List of ``{"label", "x", "y", "color"}`` dicts.
        fwhm                 : PSF FWHM in pixels (aperture diameter).
        center_coords        : ``(cy, cx)`` image centre.
        iwa                  : Inner working angle in pixels.
        statistic            : ``"mean"`` or ``"median"`` (used in title).
        save_path            : Output file path for the residual image.
        label                : Dataset/algorithm label for the title.
        draw_circles         : If True, overlay planet and noise aperture circles.
        shared_colorbar      : If True, use a fixed ±3σ colour scale.
                               If False, compute per-frame robust symmetric limits
                               so faint signals are not washed out.
        exclude_nearest      : Apertures to skip on each side of the planet in the
                               noise ring.
        exclude_other_planets: If True (default), noise apertures inside the
                               keep-out zone of any other planet are dropped.

    Returns:
        ``{"planet_label": snr_value}`` for each planet.
    """
    H, W = combined.shape

    def _other_centers(i: int) -> Optional[List[Tuple[float, float]]]:
        if not (exclude_other_planets and len(planets) > 1):
            return None
        return [(float(q["x"]), float(q["y"])) for j, q in enumerate(planets) if j != i]

    # Per-planet SNR components: {label: (snr, signal, noise_std, noise_pixels)}
    snr_components: Dict[str, tuple] = {}
    for i, p in enumerate(planets):
        snr_val, sig_val, nstd_val, npix = compute_snr_components(
            combined, p["x"], p["y"], fwhm,
            exclude_nearest=exclude_nearest,
            known_planets=_other_centers(i),
        )
        snr_components[p["label"]] = (snr_val, sig_val, nstd_val, npix)

    snr_dict: Dict[str, float] = {lbl: v[0] for lbl, v in snr_components.items()}

    if shared_colorbar:
        std = np.std(combined)
        vmin, vmax = -3 * std, 3 * std
    else:
        lo, hi = np.nanpercentile(combined, [1, 99])
        vmax = max(abs(lo), abs(hi))
        vmin = -vmax

    fig, ax = plot_image(
        combined,
        vmin=vmin,
        vmax=vmax,
        center_coords=center_coords,
        show_center=True,
        iwa_radius=iwa,
        title=None,
    )

    if draw_circles:
        for i, p in enumerate(planets):
            draw_apertures(
                ax, p["x"], p["y"], fwhm, (H, W), p["color"],
                exclude_nearest=exclude_nearest,
                known_planets=_other_centers(i),
            )

    legend_handles = []
    for p in planets:
        snr_val, sig_val, nstd_val, _ = snr_components[p["label"]]
        legend_handles.append(
            mpatches.Patch(
                color=p["color"],
                label=(
                    f"{p['label']}  SNR = {snr_val:.2f}"
                    f"  (Sig: {sig_val:.2e} / Noise: {nstd_val:.2e})"
                ),
            )
        )
    if draw_circles and exclude_other_planets and len(planets) > 1:
        r_excl = (exclude_nearest + 0.5) * fwhm
        legend_handles.append(
            mpatches.Patch(
                color="orange",
                label=f"excluded (keep-out r={r_excl:.1f} px around other planets)",
                alpha=0.8,
            )
        )
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.7)

    scale_note = "" if shared_colorbar else "  [local scale]"
    mask_note  = "" if exclude_other_planets else "  [Unmasked Noise]"
    snr_str = "  ".join(f"{p['label']}={snr_dict[p['label']]:.2f}" for p in planets)
    ax.set_title(
        f"{statistic.capitalize()} residuals — {label}  K={k}{scale_note}{mask_note}\n{snr_str}",
        fontsize=9,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Noise histograms — one per planet ────────────────────────────────────
    for p in planets:
        snr_val, sig_val, nstd_val, npix = snr_components[p["label"]]
        hist_path = save_path.parent / f"{label}_K{k}_{p['label']}_noise_hist.png"
        plot_noise_histogram(npix, p["label"], snr_val, sig_val, nstd_val, hist_path)

    return snr_dict


def plot_method_comparison(
    all_snr: Dict[str, Dict[str, Dict[int, float]]],
    k_list: List[int],
    planets: List[dict],
    output_dir: Path,
) -> None:
    """Plot SNR vs K for multiple KLIP modes on shared axes.

    Each mode is assigned a distinct line style; planet colours are preserved
    across modes so per-planet trends are visually groupable.

    Args:
        all_snr   : Nested dict ``{mode: {planet_label: {k: snr}}}``.
        k_list    : Ordered list of K values (shared across modes).
        planets   : List of ``{"label", "color"}`` dicts.
        output_dir: Directory to write the PNG.
    """
    from matplotlib.lines import Line2D

    _linestyles: Dict[str, str] = {"adi": "-", "rdi": "--", "hybrid": ":"}
    _mode_names: Dict[str, str] = {
        "adi":    "ADI    (solid)",
        "rdi":    "RDI    (dashed)",
        "hybrid": "Hybrid (dotted)",
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    for mode, snr_by_planet in all_snr.items():
        ls = _linestyles.get(mode, "-.")
        for p in planets:
            if p["label"] not in snr_by_planet:
                continue
            snr_vals = [snr_by_planet[p["label"]][k] for k in k_list]
            best_snr = max(snr_vals)
            best_k   = k_list[snr_vals.index(best_snr)]
            ax.plot(
                k_list, snr_vals,
                linestyle=ls, marker="o", linewidth=2, color=p["color"],
                label=(
                    f"{p['label']}  {mode.upper()}  "
                    f"(best K={best_k}, SNR={best_snr:.2f})"
                ),
            )
            ax.plot(best_k, best_snr, marker="*", color=p["color"],
                    markersize=14, linestyle="none")

    style_handles = [
        Line2D(
            [0], [0], color="gray",
            linestyle=_linestyles.get(m, "-."), linewidth=2,
            label=_mode_names.get(m, m.upper()),
        )
        for m in all_snr
    ]
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + style_handles, loc="best", fontsize=8, framealpha=0.7)

    modes_str = " vs ".join(m.upper() for m in all_snr)
    ax.set_title(f"{modes_str}: SNR vs K  [OS11 HLC]", fontsize=11)
    ax.set_xlabel("K  (number of KL modes)")
    ax.set_ylabel("SNR")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    fname     = "_vs_".join(m.upper() for m in all_snr) + "_SNR_curve.png"
    save_path = output_dir / fname
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison plot saved → {save_path}")

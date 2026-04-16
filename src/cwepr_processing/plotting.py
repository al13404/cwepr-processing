"""
Plotting utilities for CW-EPR processing pipeline visualization.

Requires matplotlib: pip install cwepr-processing[plot]
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_pipeline(data, n_pts=5, background_raw=None,
                  figsize=(20, 15)):
    """Plot all steps of the CW-EPR processing pipeline.

    Parameters
    ----------
    data : dict
        A single spectrum result from process_directory()
        or process_spectrum(), must also contain 'field'
        and 'raw' keys (as returned by process_directory).
    n_pts : int
        Number of edge points used for anchor display
        (default 5, should match baseline_points).
    background_raw : array or None
        Raw background trace for display in the first
        panel. If None, panel shows "no background."
    figsize : tuple
        Figure size (default (20, 15)).

    Returns
    -------
    fig, axes
        Matplotlib figure and 3x3 axes array.
    """
    field = data["field"]
    rc = data["residual_correction"]
    condition = data.get("condition", "spectrum")

    # Compute anchor points for quadratic baseline display
    x_start = np.mean(field[:n_pts])
    x_end = np.mean(field[-n_pts:])
    x_center = np.mean(field)
    x_anchors = [x_start, x_center, x_end]
    y_anchors = np.polyval(
        np.polyfit(
            [x_start, x_center, x_end],
            [np.mean(data["after_bg"][:n_pts]),
             data["y_center"],
             np.mean(data["after_bg"][-n_pts:])],
            2,
        ),
        x_anchors,
    )

    fig, axes = plt.subplots(3, 3, figsize=figsize)

    # ---- Row 0: Background + baseline ----

    # Panel 0,0: Background
    if background_raw is not None:
        axes[0, 0].plot(
            field, background_raw,
            "r-", lw=1, alpha=0.6,
            label="Raw background",
        )
        axes[0, 0].plot(
            field,
            data["bg_scale"] * background_raw,
            "k--", lw=2,
            label=f"Scaled (x{data['bg_scale']:.4f})",
        )
        axes[0, 0].axhline(0, color="gray",
                            ls="--", alpha=0.5)
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].set_title(
            "Step 0: Background\n"
            f"(scale = {data['bg_scale']:.4f},"
            " optimized)"
        )
    else:
        axes[0, 0].text(
            0.5, 0.5,
            "No background\nspectrum provided",
            transform=axes[0, 0].transAxes,
            ha="center", va="center",
            fontsize=14, color="gray",
        )
        axes[0, 0].set_title("Step 0: Background")
    axes[0, 0].set_xlabel("Magnetic Field (G)")
    axes[0, 0].set_ylabel("Intensity (a.u.)")

    # Panel 0,1: BG subtraction + quadratic fit
    axes[0, 1].plot(field, data["raw"],
                    "b-", lw=1, alpha=0.7,
                    label="Raw spectrum")
    if background_raw is not None:
        axes[0, 1].plot(
            field,
            data["bg_scale"] * background_raw,
            "r-", lw=1.5,
            label=f"Scaled BG (x{data['bg_scale']:.4f})",
        )
    axes[0, 1].plot(field, data["after_bg"],
                    "g-", lw=1,
                    label="After BG subtraction")
    axes[0, 1].plot(field, data["baseline_quad"],
                    "m-", lw=1.5,
                    label="Quadratic baseline")
    axes[0, 1].plot(x_anchors, y_anchors,
                    "mo", ms=8, zorder=5,
                    label="Anchor points")
    axes[0, 1].axhline(0, color="gray",
                        ls="--", alpha=0.5)
    axes[0, 1].set_title(
        "Step 1: BG Subtraction + Quadratic Fit"
    )
    axes[0, 1].set_xlabel("Magnetic Field (G)")
    axes[0, 1].set_ylabel("Intensity (a.u.)")
    axes[0, 1].legend(fontsize=7)

    # Panel 0,2: Detrended derivative
    axes[0, 2].plot(field, data["detrended"],
                    "b-", lw=1)
    axes[0, 2].axhline(0, color="gray",
                        ls="--", alpha=0.5)
    axes[0, 2].set_title(
        "Step 2: Detrended Derivative\n"
        "(after BG + quadratic removal)"
    )
    axes[0, 2].set_xlabel("Magnetic Field (G)")
    axes[0, 2].set_ylabel("Intensity (a.u.)")

    # ---- Row 1: Residual correction pass ----

    # Panel 1,0: Absorption with residual baseline
    axes[1, 0].plot(
        field, rc["absorption_raw"],
        "r-", lw=1,
        label="Absorption (1st integral)",
    )
    axes[1, 0].plot(
        field, rc["residual_baseline"],
        "k--", lw=1.5,
        label=(f"Residual baseline\n"
               f"(first/last {n_pts} pts)"),
    )
    axes[1, 0].axhline(0, color="gray",
                        ls="--", alpha=0.3)
    axes[1, 0].set_title(
        "Step 3: Absorption\n"
        "with residual baseline"
    )
    axes[1, 0].set_xlabel("Magnetic Field (G)")
    axes[1, 0].set_ylabel("Absorption (a.u.)")
    axes[1, 0].legend(fontsize=8)

    # Panel 1,1: Corrected absorption
    axes[1, 1].plot(
        field, rc["absorption_offset"],
        "r-", lw=1.2,
    )
    axes[1, 1].fill_between(
        field, 0, rc["absorption_offset"],
        alpha=0.15, color="red",
    )
    axes[1, 1].axhline(0, color="gray",
                        ls="--", alpha=0.3)
    axes[1, 1].set_title(
        "Step 4: Corrected Absorption\n"
        "(baseline removed, offset >= 0)"
    )
    axes[1, 1].set_xlabel("Magnetic Field (G)")
    axes[1, 1].set_ylabel("Absorption (a.u.)")

    # Panel 1,2: Corrected first-derivative
    axes[1, 2].plot(
        field, rc["corrected_derivative"],
        "b-", lw=1,
    )
    axes[1, 2].axhline(0, color="gray",
                        ls="--", alpha=0.5)
    axes[1, 2].set_title(
        "Step 5: Corrected First-Derivative"
    )
    axes[1, 2].set_xlabel("Magnetic Field (G)")
    axes[1, 2].set_ylabel("Intensity (a.u.)")

    # ---- Row 2: Normalization ----

    # Panel 2,0: Double integral + normalized
    ax_cum = axes[2, 0]
    ax_norm = ax_cum.twinx()
    ax_cum.plot(field, data["di_cumulative"],
                "g-", lw=1.2)
    ax_cum.axhline(
        data["double_integral"],
        color="darkgreen", ls="--", alpha=0.7,
    )
    ax_cum.text(
        field[len(field) // 4],
        data["double_integral"] * 0.92,
        f"Total = {data['double_integral']:.4g}",
        fontsize=10, color="darkgreen",
    )
    ax_cum.set_xlabel("Magnetic Field (G)")
    ax_cum.set_ylabel("Cumulative Area", color="g")
    ax_norm.plot(field, data["normalized"],
                 "purple", lw=1.2)
    ax_norm.set_ylabel("Norm. Intensity",
                        color="purple")
    ax_cum.set_title(
        "Step 6: Double Integral &\n"
        "Area-Normalized Spectrum"
    )

    # Panel 2,1: Summary text
    axes[2, 1].axis("off")
    summary_lines = [
        f"Condition: {condition}",
        "",
        f"Background scale: {data['bg_scale']:.6f}",
        f"Optimizer cost: {data['cost']:.4e}",
        f"Converged: {data['converged']}",
        f"y_center (opt): {data['y_center']:.4g}",
        f"Double integral: {data['double_integral']:.4g}",
    ]
    axes[2, 1].text(
        0.1, 0.9, "\n".join(summary_lines),
        transform=axes[2, 1].transAxes,
        fontsize=12, verticalalignment="top",
        fontfamily="monospace",
    )
    axes[2, 1].set_title("Processing Summary")

    # Panel 2,2: empty
    axes[2, 2].axis("off")

    fig.suptitle(
        f"Processing Pipeline: {condition}",
        fontsize=14,
    )
    plt.tight_layout()

    return fig, axes

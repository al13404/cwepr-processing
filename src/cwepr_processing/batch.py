"""
Batch processing for CW-EPR datasets.

Provides process_directory() which automates the full workflow:
find files, detect background, process, normalize.
"""

from __future__ import annotations

from pathlib import Path

from .io import (find_all_epr_files, find_background_file,
                 load_epr_data)
from .processing import process_spectrum


def extract_condition(filename: str) -> str:
    """Extract a condition label from a filename stem.

    Simply returns the filename stem as-is, which works
    universally regardless of naming convention.
    """
    return filename


def process_directory(
    base_dir: str | None = None,
    *,
    subtract_background: bool = True,
    baseline_points: int = 5,
    normalization: str = "area",
    verbose: bool = True,
) -> dict:
    """Process all CW-EPR spectra in a directory.

    Parameters
    ----------
    base_dir : str or None
        Path to directory with .DSC/.DTA files. Prompts if None.
    subtract_background : bool
        Look for and subtract a Background file (default True).
    baseline_points : int
        Edge points for baseline anchors/cost (default 5).
    normalization : str
        'area' (default), 'amplitude', or 'none'.
    verbose : bool
        Print progress (default True).

    Returns
    -------
    dict
        Keyed by filename stem. Each value contains field, raw,
        condition, params, and all process_spectrum outputs.
    """
    if base_dir is None:
        base_dir = input(
            "Enter path to CW-EPR data directory: "
        ).strip()

    base_dir_path = Path(base_dir)
    if not base_dir_path.is_dir():
        raise FileNotFoundError(
            f"Directory not found: {base_dir}"
        )

    # Find background
    background_intensity = None
    if subtract_background:
        bg_file = find_background_file(base_dir)
        if bg_file is not None:
            bg_field, bg_intensity, bg_params = (
                load_epr_data(bg_file)
            )
            background_intensity = bg_intensity
            if verbose:
                xmin = bg_params['XMIN']
                xwid = bg_params['XWID']
                print(
                    f"  Background loaded:"
                    f" {Path(bg_file).name}\n"
                    f"    Points: {bg_params['XPTS']}\n"
                    f"    Field range:"
                    f" {xmin:.1f} -- {xmin + xwid:.1f} G"
                )
        else:
            if verbose:
                print("  No background file found --"
                      " proceeding without.")

    # Discover data files
    epr_files = find_all_epr_files(base_dir)
    if not epr_files:
        raise ValueError(
            f"No EPR data files found in {base_dir}"
        )

    if verbose:
        print(f"\n  Found {len(epr_files)}"
              f" EPR data file(s).\n")

    # Process each spectrum
    all_results: dict = {}

    for filepath, filename in epr_files:
        try:
            field, intensity, params = (
                load_epr_data(filepath)
            )
            condition = extract_condition(filename)

            result = process_spectrum(
                field, intensity,
                background_raw=background_intensity,
                n_pts=baseline_points,
            )

            # Optional amplitude normalization override
            if normalization == "amplitude":
                amp = (
                    result["residual_correction"]["corrected_derivative"].max()
                    - result["residual_correction"]["corrected_derivative"].min()
                )
                if abs(amp) > 0:
                    result["normalized"] = (
                        result["residual_correction"]["corrected_derivative"]
                        / amp
                    )
            elif normalization == "none":
                result["normalized"] = (
                    result["residual_correction"]["corrected_derivative"]
                )

            all_results[filename] = {
                "field": field,
                "raw": intensity,
                "condition": condition,
                "params": params,
                **result,
            }

            if verbose:
                status = "\u2713" if result["converged"] else "\u2717"
                print(
                    f"  {status} {condition:24s}  "
                    f"bg_scale={result['bg_scale']:.4f}  "
                    f"cost={result['cost']:.2e}  "
                    f"DI={result['double_integral']:.4g}"
                )

        except Exception as exc:
            if verbose:
                print(
                    f"  \u2717 Error processing"
                    f" {filename}: {exc}"
                )

    return all_results

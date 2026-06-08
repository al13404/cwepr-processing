# cwepr-processing

**CW-EPR data processing toolkit for Bruker DSC/DTA and Magnettech CSV files.**

A reusable Python package for loading, processing, and normalizing
continuous-wave EPR spectra from Bruker `.DSC`/`.DTA` files and
Magnettech (ESR5000 / MS-5000) `.csv` exports.

## Features

- **Multi-format file I/O**
  - Bruker `.DSC`/`.DTA` binary files
  - Magnettech semicolon-delimited `.csv` exports (field auto-converted
    from mT to Gauss)
  - Auto-detection -- `process_directory()` handles mixed directories
    seamlessly
- **Automatic background detection** -- Finds any file with "background"
  in the name (case-insensitive)
- **Full processing pipeline:**
  1. Background subtraction (jointly optimized scaling factor)
  2. Quadratic baseline correction (3-point anchor fit)
  3. Detrending
  4. Integration to absorption (trapezoidal)
  5. Joint optimization of bg scale and baseline curvature
  6. Residual linear baseline correction in absorption domain
  7. Area or amplitude normalization
- **Batch processing** -- Process an entire directory with one call
- **Pipeline visualization** -- Built-in plotting of every processing step

## Installation

```bash
pip install git+https://github.com/al13404/cwepr-processing.git
```

To also pull in `matplotlib` for the plotting helpers:

```bash
pip install "cwepr-processing[plot] @ git+https://github.com/al13404/cwepr-processing.git"
```

### For development (editable install)

```bash
git clone https://github.com/al13404/cwepr-processing.git
cd cwepr-processing
pip install -e ".[dev]"
```

## Quick Start

```python
from cwepr_processing import process_directory

results = process_directory(r"C:\path\to\your\epr\data")

for name, data in results.items():
    print(f"{data['condition']:20s}  DI={data['double_integral']:.4g}")
```

`results` is a dict keyed by filename stem. Each entry contains the
field axis, the raw and processed traces, and every diagnostic the
pipeline computed -- see [`process_spectrum`](#processing----cwepr_processingprocessing)
below for the full key list.

## Plotting Your Processed Data

Plotting requires `matplotlib`. Install with `pip install matplotlib`
or use the `[plot]` extra shown above.

### Overlay normalized spectra

The simplest way to compare samples: loop over the result dict and
plot each `normalized` trace against `field`.

```python
import matplotlib.pyplot as plt
from cwepr_processing import process_directory

results = process_directory(r"C:\path\to\your\epr\data")

fig, ax = plt.subplots(figsize=(8, 5))
for name, data in results.items():
    ax.plot(data["field"], data["normalized"],
            label=data["condition"], lw=1.2)

ax.set_xlabel("Magnetic field (G)")
ax.set_ylabel("Normalized intensity (a.u.)")
ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
```

Swap `data["normalized"]` for `data["di_cumulative"]` to compare the
cumulative double integrals, or for `data["residual_correction"]["absorption_offset"]`
to plot the corrected absorption spectra.

### Full diagnostic figure with `plot_pipeline`

To see every processing step for a single spectrum (background, quadratic
fit, detrended derivative, absorption, residual correction, double integral
and the area-normalized result), use `plot_pipeline`:

```python
import matplotlib.pyplot as plt
from cwepr_processing import (
    process_directory, load_any_epr, find_background_file, plot_pipeline,
)

data_dir = r"C:\path\to\your\epr\data"
results = process_directory(data_dir)

# Optional: load the background trace so it appears in the first panel
bg_file = find_background_file(data_dir)
bg_intensity = None
if bg_file is not None:
    _, bg_intensity, _ = load_any_epr(bg_file)

# Pick any spectrum from the results dict
sample = next(iter(results.values()))

fig, axes = plot_pipeline(sample, n_pts=5,
                          background_raw=bg_intensity)
plt.show()
```

`plot_pipeline` returns the matplotlib `(fig, axes)` so you can keep
customizing -- for example `fig.savefig("diagnostic.png", dpi=200)`.

## Processing Flow

1. **File I/O** -- Bruker binary `.DTA` spectral data and `.DSC` parameter
   files are parsed to extract the raw first-derivative intensity and the
   magnetic field axis.  Magnettech `.csv` exports are read directly from the
   semicolon-delimited text; the native mT field values are converted to
   Gauss (x10) so that all downstream processing uses a consistent unit.
   Format detection is automatic.

2. **Background Detection & Subtraction** — If a file with "background" in its
   name is present, it is automatically identified and subtracted from each
   spectrum. The background scaling factor is not fixed at 1.0 — it is jointly
   optimized in step 5.

3. **Quadratic Baseline Correction** — A 3-point quadratic polynomial is fit
   through anchor points at the start, center, and end of the spectrum
   (averaged over `n_pts` edge points). This removes broad curvature from the
   derivative signal.

4. **Detrending** — The fitted quadratic baseline is subtracted, yielding a
   corrected first-derivative spectrum.

5. **Joint Optimization** — The background scale factor and the center anchor
   value of the quadratic baseline are simultaneously optimized using L-BFGS-B
   minimization. The cost function minimizes the squared mean intensity at both
   edges of the integrated (absorption) spectrum — driving the absorption
   baseline toward zero at the spectral boundaries.

6. **Integration to Absorption** — The corrected derivative is numerically
   integrated (trapezoidal rule) to produce the absorption spectrum.

7. **Residual Linear Correction** — Any remaining linear drift in the
   absorption domain is removed by a second-pass detrend-and-baseline step,
   ensuring the absorption spectrum starts and ends near zero.

8. **Normalization** — The final derivative spectrum is normalized by the
   double integral (total spectral area), enabling quantitative comparison of
   spin concentration across samples. Amplitude normalization is also available.

The `process_directory()` function automates this entire pipeline across all
spectra in a folder, and `plot_pipeline()` generates a 3×3 diagnostic figure
showing every intermediate step for visual verification.

## API Reference

Everything below is exported from the top-level `cwepr_processing`
namespace, so `from cwepr_processing import process_directory` is
equivalent to `from cwepr_processing.batch import process_directory`.

### I/O -- `cwepr_processing.io`

#### `read_dsc_file(dsc_path) -> dict`
Parse a Bruker `.DSC` parameter file. Returns a dict with keys such as
`XPTS`, `XMIN`, `XWID`, `TITLE`, `BSEQ`, `IKKF`, `XUNI`.

#### `read_dta_file(dta_path, num_points, byte_order="big") -> ndarray`
Read a Bruker `.DTA` binary file. Returns a NumPy array of intensities.
`byte_order` is `"big"` or `"little"` -- normally set automatically by
`load_bruker_data` based on the `BSEQ` field in the `.DSC`.

#### `read_magnettech_csv(csv_path) -> dict`
Parse a Magnettech semicolon-delimited `.csv` export. Returns
`{"field_mT": ndarray, "intensity": ndarray, "params": dict}` where
`params` carries the full metadata header (recipe settings, MW
frequency, Q-factor, temperature, accumulations, etc.).

#### `load_bruker_data(dsc_path) -> (field, intensity, params)`
High-level Bruker loader. Reads the `.DSC`/`.DTA` pair, reconstructs
the field axis from `XMIN`/`XWID`/`XPTS`, and returns the field (Gauss),
the intensity, and the parameter dict (with `SOURCE="bruker"` added).

#### `load_epr_data(dsc_path) -> (field, intensity, params)`
Backwards-compatible alias for `load_bruker_data`.

#### `load_magnettech_data(csv_path, field_unit="G") -> (field, intensity, params)`
High-level Magnettech loader. Returns the same `(field, intensity,
params)` tuple as the Bruker loader. Set `field_unit="mT"` to keep
the native units; the default `"G"` multiplies the field axis by 10.
`params["SOURCE"]` is set to `"magnettech"`.

#### `load_any_epr(filepath, **kwargs) -> (field, intensity, params)`
Auto-detect format from extension and a quick content sniff, then
dispatch to the right loader. Extra `**kwargs` are forwarded to the
underlying loader (e.g. `field_unit="mT"`).

#### `find_all_epr_files(base_dir) -> list[tuple[str, str]]`
Scan a directory for every loadable `.DSC`/`.DTA` pair and Magnettech
`.csv`, skipping anything with "background" in the name. Returns a
sorted list of `(filepath, filename_stem)` tuples.

#### `find_background_file(base_dir) -> str | None`
Return the path of the first file in `base_dir` whose name contains
"background" (case-insensitive) and which is a loadable Bruker pair
or Magnettech CSV. Returns `None` if no background is present.

### Processing -- `cwepr_processing.processing`

#### `process_spectrum(field, raw_derivative, background_raw=None, n_pts=10) -> dict`
Run the full processing pipeline on a single spectrum -- background
subtraction, quadratic baseline correction, detrending, integration,
joint L-BFGS-B optimization of background scale and baseline curvature,
residual linear correction, and area normalization.

The returned dict contains:

| Key                         | Description                                                                                                    |
| --------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `after_bg`                  | Derivative after background subtraction                                                                        |
| `baseline_quad`             | The fitted quadratic baseline                                                                                  |
| `detrended`                 | Derivative after baseline removal                                                                              |
| `absorption_opt`            | Absorption integrated from the optimizer pass                                                                  |
| `residual_correction`       | Sub-dict (see below) with the second-pass results                                                              |
| `di_cumulative`             | Cumulative double-integral array                                                                               |
| `double_integral`           | Total spectral area (scalar)                                                                                   |
| `normalized`                | Area-normalized derivative spectrum                                                                            |
| `bg_scale`                  | Optimized background scale factor (`0.0` if no background)                                                     |
| `y_center`                  | Optimized quadratic baseline center anchor                                                                     |
| `cost`                      | Final optimizer cost                                                                                           |
| `converged`                 | Bool -- did the optimizer report success                                                                       |

`residual_correction` is itself a dict with keys: `detrend_line`,
`derivative_detrended`, `absorption_raw`, `residual_baseline`,
`absorption_corrected`, `absorption_offset`, `corrected_derivative`.

### Batch processing -- `cwepr_processing.batch`

#### `process_directory(base_dir=None, *, subtract_background=True, baseline_points=5, normalization="area", verbose=True) -> dict`
Run `process_spectrum` over every EPR file in `base_dir`. If `base_dir`
is `None`, you'll be prompted at the console.

Parameters:

- `subtract_background` -- look for and subtract a file matching
  `*background*` (default `True`).
- `baseline_points` -- number of edge points used for the anchor
  averages and the optimizer's edge cost (default `5`).
- `normalization` -- `"area"` (default), `"amplitude"`, or `"none"`.
- `verbose` -- print per-spectrum progress (default `True`).

Returns a dict keyed by filename stem. Each value contains `field`,
`raw`, `condition`, `params`, and every key from `process_spectrum`.

#### `extract_condition(filename) -> str`
Returns the filename stem as-is. This is the hook point if you want
to derive a custom condition label from a filename -- subclass or
monkey-patch as needed.

### Plotting -- `cwepr_processing.plotting`

Requires `matplotlib`.

#### `plot_pipeline(data, n_pts=5, background_raw=None, figsize=(20, 15)) -> (fig, axes)`
Generate a 3×3 diagnostic figure showing every stage of the pipeline:
background trace, background subtraction with the quadratic fit,
detrended derivative, absorption with the residual baseline, corrected
absorption, corrected derivative, double-integral overlay with the
area-normalized spectrum, and a text summary panel.

`data` is a single result entry from `process_directory()` (or any
dict matching the `process_spectrum` output with `field` and `raw`
added). Pass `background_raw` to populate the background panel --
otherwise it shows a "no background" placeholder.

Returns the matplotlib `(fig, axes)` so you can save or further
customize the figure.

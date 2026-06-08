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

## Installation

```bash
pip install git+https://github.com/al13404/cwepr-processing.git
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

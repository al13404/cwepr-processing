# cwepr-processing

**CW-EPR data processing toolkit for Bruker DSC/DTA files.**

A reusable Python package for loading, processing, and normalizing
continuous-wave EPR spectra from Bruker `.DSC`/`.DTA` files.

## Features

- **Bruker file I/O** -- Read `.DSC` parameter files and `.DTA` binary data
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

1. **File I/O** — Binary `.DTA` spectral data and `.DSC` parameter files are
   parsed to extract the raw first-derivative intensity and the magnetic field
   axis (start field, sweep width, number of points, byte order).

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
pip install git+https://github.com/amg43-mcw/cwepr-processing.git
```

### For development (editable install)

```bash
git clone https://github.com/amg43-mcw/cwepr-processing.git
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

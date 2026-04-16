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

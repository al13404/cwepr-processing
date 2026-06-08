"""
cwepr_processing -- CW-EPR data processing toolkit.

Quick start:
    >>> from cwepr_processing import process_directory
    >>> results = process_directory("/path/to/data")
"""

from .io import (
    find_all_epr_files,
    find_background_file,
    load_any_epr,
    load_bruker_data,
    load_epr_data,
    load_magnettech_data,
    read_dsc_file,
    read_dta_file,
    read_magnettech_csv,
)
from .processing import process_spectrum
from .batch import extract_condition, process_directory
from .plotting import plot_pipeline

__version__ = "0.2.0"

__all__ = [
    "read_dsc_file",
    "read_dta_file",
    "read_magnettech_csv",
    "load_epr_data",
    "load_bruker_data",
    "load_magnettech_data",
    "load_any_epr",
    "find_all_epr_files",
    "find_background_file",
    "process_spectrum",
    "extract_condition",
    "process_directory",
    "plot_pipeline",
]

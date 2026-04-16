"""
Bruker DSC/DTA file I/O utilities for CW-EPR data.
"""

import re
import struct
from pathlib import Path

import numpy as np


def _is_background_file(filename_stem: str) -> bool:
    """Check if a filename contains 'background' (case-insensitive)."""
    return "background" in filename_stem.lower()


def read_dsc_file(dsc_path: str) -> dict:
    """Read a Bruker DSC parameter file and extract field sweep parameters."""
    params = {}
    with open(dsc_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("XPTS"):
                params["XPTS"] = int(line.split()[1])
            elif line.startswith("XMIN"):
                params["XMIN"] = float(line.split()[1])
            elif line.startswith("XWID"):
                params["XWID"] = float(line.split()[1])
            elif line.startswith("TITL"):
                match = re.search(r"'(.+?)'", line)
                if match:
                    params["TITLE"] = match.group(1)
            elif line.startswith("BSEQ"):
                params["BSEQ"] = line.split()[1]
            elif line.startswith("IKKF"):
                params["IKKF"] = line.split()[1]
            elif line.startswith("XUNI"):
                match = re.search(r"'(.+?)'", line)
                if match:
                    params["XUNI"] = match.group(1)
    return params


def read_dta_file(
    dta_path: str, num_points: int, byte_order: str = "big"
) -> np.ndarray:
    """Read a Bruker DTA binary data file."""
    with open(dta_path, "rb") as f:
        data = f.read()
    num_values = len(data) // 8
    fmt = f">{num_values}d" if byte_order == "big" else f"<{num_values}d"
    values = struct.unpack(fmt, data)
    return np.array(values[:num_points])


def load_epr_data(dsc_path: str) -> tuple:
    """Load EPR data from a DSC/DTA file pair.

    Returns (field, intensity, params).
    """
    dta_path = dsc_path.replace(".DSC", ".DTA")
    params = read_dsc_file(dsc_path)
    field = np.linspace(
        params["XMIN"], params["XMIN"] + params["XWID"], params["XPTS"]
    )
    byte_order = "big" if params.get("BSEQ", "BIG") == "BIG" else "little"
    intensity = read_dta_file(dta_path, params["XPTS"], byte_order)
    return field, intensity, params


def find_all_epr_files(base_dir: str) -> list:
    """Find all DSC/DTA file pairs in base_dir, excluding background files.

    Any file with 'background' anywhere in its name (case-insensitive)
    is excluded.
    """
    base_dir = Path(base_dir)
    epr_files = []
    for file in sorted(base_dir.iterdir()):
        if file.suffix.upper() == ".DSC":
            if _is_background_file(file.stem):
                continue
            if file.with_suffix(".DTA").exists():
                epr_files.append((str(file), file.stem))
    return epr_files


def find_background_file(base_dir: str):
    """Find the background .DSC file in base_dir.

    Matches any file with 'background' anywhere in its name
    (case-insensitive) that has a matching .DTA file.

    Returns path or None.
    """
    base_dir = Path(base_dir)
    for file in sorted(base_dir.iterdir()):
        if file.suffix.upper() == ".DSC":
            if _is_background_file(file.stem):
                if file.with_suffix(".DTA").exists():
                    return str(file)
    return None

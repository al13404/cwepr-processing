"""
File I/O utilities for CW-EPR data.

Supported formats
-----------------
- **Bruker DSC/DTA** -- Binary spectral data (.DTA) with parameter
  sidecar (.DSC).  Field axis is reconstructed from XMIN/XWID/XPTS.
- **Magnettech CSV** -- Semicolon-delimited text files exported by
  Magnettech ESR5000 / MS-5000 software.  Contains a metadata header
  followed by a ``Meas`` section with ``BField [mT]`` and
  ``MW_Absorption`` columns.  Field values are converted from mT to
  Gauss on load so that all downstream processing uses a consistent
  unit.
"""

import re
import struct
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ #
#  Shared helpers                                                      #
# ------------------------------------------------------------------ #

def _is_background_file(filename_stem: str) -> bool:
    """Check if a filename contains 'background' (case-insensitive)."""
    return "background" in filename_stem.lower()


def _is_magnettech_csv(filepath: str) -> bool:
    """Return True if *filepath* looks like a Magnettech CSV export.

    Detection is intentionally conservative: the file must contain a
    ``Meas`` section marker and a ``BField`` column header somewhere
    in the first 120 lines.
    """
    try:
        with open(filepath, "r", encoding="utf-8-sig") as fh:
            for i, line in enumerate(fh):
                if i > 120:
                    break
                stripped = line.strip()
                if stripped.startswith("BField"):
                    return True
    except (UnicodeDecodeError, OSError):
        pass
    return False


# ------------------------------------------------------------------ #
#  Bruker DSC / DTA                                                    #
# ------------------------------------------------------------------ #

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


def load_bruker_data(dsc_path: str) -> tuple:
    """Load EPR data from a Bruker DSC/DTA file pair.

    Returns (field_G, intensity, params).
    """
    dta_path = dsc_path.replace(".DSC", ".DTA")
    params = read_dsc_file(dsc_path)
    field = np.linspace(
        params["XMIN"], params["XMIN"] + params["XWID"], params["XPTS"]
    )
    byte_order = "big" if params.get("BSEQ", "BIG") == "BIG" else "little"
    intensity = read_dta_file(dta_path, params["XPTS"], byte_order)
    params["SOURCE"] = "bruker"
    return field, intensity, params


# keep the old name as an alias for backwards compatibility
load_epr_data = load_bruker_data


# ------------------------------------------------------------------ #
#  Magnettech CSV                                                      #
# ------------------------------------------------------------------ #

def read_magnettech_csv(csv_path: str) -> dict:
    """Read a Magnettech CSV export and return metadata + raw arrays.

    Parameters
    ----------
    csv_path : str or Path
        Path to the ``.csv`` file.

    Returns
    -------
    dict with keys:
        ``field_mT``   -- numpy array, magnetic field in mT
        ``intensity``  -- numpy array, MW absorption signal
        ``params``     -- dict of parsed metadata (recipe settings,
                          measurement info, MW frequency, etc.)
    """
    params: dict = {}
    field_vals: list = []
    intensity_vals: list = []
    in_data = False

    with open(csv_path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip().replace("\r", "")

            # Data section: two semicolon-separated floats per line
            if in_data:
                parts = line.split(";")
                if len(parts) == 2:
                    try:
                        field_vals.append(float(parts[0]))
                        intensity_vals.append(float(parts[1]))
                    except ValueError:
                        pass
                continue

            # Column header marks the start of data on the NEXT line
            if line.startswith("BField"):
                in_data = True
                continue

            # Metadata lines: ``key;value;description`` or ``key;value``
            parts = line.split(";")
            if len(parts) >= 2 and parts[0]:
                key = parts[0].strip()
                val = parts[1].strip()
                if val:
                    # Try to cast numeric values
                    try:
                        val = int(val)
                    except ValueError:
                        try:
                            val = float(val)
                        except ValueError:
                            pass
                    params[key] = val

    if not field_vals:
        raise ValueError(
            f"No spectral data found in {csv_path}"
        )

    return {
        "field_mT": np.array(field_vals),
        "intensity": np.array(intensity_vals),
        "params": params,
    }


def load_magnettech_data(
    csv_path: str,
    field_unit: str = "G",
) -> tuple:
    """Load EPR data from a Magnettech CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the Magnettech ``.csv`` export.
    field_unit : str
        Unit for the returned field axis.  ``'G'`` (default) converts
        the native mT values to Gauss (x10).  ``'mT'`` keeps the
        original values.

    Returns
    -------
    (field, intensity, params)
        Same signature as :func:`load_bruker_data` / ``load_epr_data``.
    """
    raw = read_magnettech_csv(csv_path)
    params = raw["params"]
    field = raw["field_mT"].copy()

    if field_unit == "G":
        field *= 10.0  # 1 mT = 10 G
        params["XUNI"] = "G"
    else:
        params["XUNI"] = "mT"

    params["XPTS"] = len(field)
    params["XMIN"] = float(field[0])
    params["XWID"] = float(field[-1] - field[0])
    params["SOURCE"] = "magnettech"

    title = params.get("Name", Path(csv_path).stem)
    params["TITLE"] = title

    return field, raw["intensity"], params


# ------------------------------------------------------------------ #
#  Unified loader                                                      #
# ------------------------------------------------------------------ #

def load_any_epr(filepath: str, **kwargs) -> tuple:
    """Auto-detect file format and load EPR data.

    Dispatches to :func:`load_bruker_data` for ``.DSC`` files and
    :func:`load_magnettech_data` for Magnettech ``.csv`` files.

    Returns (field, intensity, params).
    """
    p = Path(filepath)
    if p.suffix.upper() == ".DSC":
        return load_bruker_data(filepath)
    if p.suffix.lower() == ".csv" and _is_magnettech_csv(filepath):
        return load_magnettech_data(filepath, **kwargs)
    raise ValueError(
        f"Unsupported file format: {p.suffix!r} -- "
        "expected .DSC (Bruker) or Magnettech .csv"
    )


# ------------------------------------------------------------------ #
#  Directory scanning                                                  #
# ------------------------------------------------------------------ #

def find_all_epr_files(base_dir: str) -> list:
    """Find all EPR data files in *base_dir*, excluding backgrounds.

    Discovers both Bruker DSC/DTA pairs and Magnettech CSV exports.
    Any file with 'background' anywhere in its name (case-insensitive)
    is excluded.
    """
    base_dir = Path(base_dir)
    epr_files = []

    for file in sorted(base_dir.iterdir()):
        if _is_background_file(file.stem):
            continue

        # Bruker DSC/DTA pair
        if file.suffix.upper() == ".DSC":
            if file.with_suffix(".DTA").exists():
                epr_files.append((str(file), file.stem))

        # Magnettech CSV
        elif file.suffix.lower() == ".csv":
            if _is_magnettech_csv(str(file)):
                epr_files.append((str(file), file.stem))

    return epr_files


def find_background_file(base_dir: str):
    """Find a background EPR file in *base_dir*.

    Matches any file with 'background' anywhere in its name
    (case-insensitive) that is a valid Bruker DSC/DTA pair or a
    Magnettech CSV.

    Returns path or None.
    """
    base_dir = Path(base_dir)
    for file in sorted(base_dir.iterdir()):
        if not _is_background_file(file.stem):
            continue

        if file.suffix.upper() == ".DSC":
            if file.with_suffix(".DTA").exists():
                return str(file)

        elif file.suffix.lower() == ".csv":
            if _is_magnettech_csv(str(file)):
                return str(file)

    return None

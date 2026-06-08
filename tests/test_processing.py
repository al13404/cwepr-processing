"""Basic tests for cwepr_processing."""

import numpy as np
import pytest
import tempfile
import os

from cwepr_processing.processing import process_spectrum
from cwepr_processing.io import (
    _is_magnettech_csv,
    read_magnettech_csv,
    load_magnettech_data,
    load_any_epr,
    find_all_epr_files,
    find_background_file,
)


def _make_synthetic_lorentzian(
    n_pts=1024, center=3467.0,
    width=100.0, linewidth=2.0,
):
    """Synthetic Lorentzian first-derivative."""
    field = np.linspace(
        center - width / 2,
        center + width / 2, n_pts,
    )
    gamma = linewidth / 2
    absorption = 1.0 / (
        1.0 + ((field - center) / gamma) ** 2
    )
    derivative = np.gradient(absorption, field)
    return field, derivative


def _write_magnettech_csv(path, field_mT, intensity,
                          name="test_spectrum"):
    """Write a minimal Magnettech-style CSV for testing."""
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write(f"Name;{name}\r\n")
        f.write("\r\n")
        f.write("Recipe\r\n")
        f.write("Accumulations;10;Number of accumulations\r\n")
        f.write("Bfrom;332;B from\r\n")
        f.write("Bto;342;B to\r\n")
        f.write("MicrowavePower;36.3;Microwave power\r\n")
        f.write("\r\n")
        f.write("Frequency;9.468\r\n")
        f.write("QFactor;1571.2\r\n")
        f.write("Temperature;25.0\r\n")
        f.write("\r\n")
        f.write("Meas\r\n")
        f.write("BField [mT];MW_Absorption []\r\n")
        for b, y in zip(field_mT, intensity):
            f.write(f"{b};{y}\r\n")


class TestProcessSpectrum:
    """Tests for the process_spectrum function."""

    def test_no_background(self):
        field, derivative = _make_synthetic_lorentzian()
        result = process_spectrum(
            field, derivative,
            background_raw=None, n_pts=5,
        )
        assert result["converged"]
        assert result["bg_scale"] == 0.0
        assert result["double_integral"] > 0

    def test_with_background(self):
        field, derivative = _make_synthetic_lorentzian()
        rng = np.random.default_rng(42)
        bg = rng.normal(0, 0.001, size=len(field))
        noisy = derivative + bg
        result = process_spectrum(
            field, noisy, background_raw=bg, n_pts=5,
        )
        assert result["converged"]
        assert 0.5 <= result["bg_scale"] <= 2.0

    def test_output_shapes(self):
        n = 512
        field, derivative = _make_synthetic_lorentzian(
            n_pts=n,
        )
        result = process_spectrum(
            field, derivative, n_pts=5,
        )
        assert result["after_bg"].shape == (n,)
        assert result["baseline_quad"].shape == (n,)
        assert result["detrended"].shape == (n,)
        assert result["normalized"].shape == (n,)
        assert result["di_cumulative"].shape == (n,)

    def test_normalized_integral_near_one(self):
        field, derivative = _make_synthetic_lorentzian()
        result = process_spectrum(
            field, derivative, n_pts=5,
        )
        dx = np.diff(field)
        avg = 0.5 * (
            result["normalized"][:-1]
            + result["normalized"][1:]
        )
        integral = np.sum(avg * dx)
        assert abs(integral) < 5.0


class TestMagnettechIO:
    """Tests for Magnettech CSV I/O."""

    def _make_test_csv(self, tmpdir, name="test_spectrum"):
        """Create a test Magnettech CSV with synthetic data."""
        field_mT = np.linspace(332, 342, 256)
        center_mT = 337.0
        gamma = 0.2
        absorption = 1.0 / (
            1.0 + ((field_mT - center_mT) / gamma) ** 2
        )
        derivative = np.gradient(absorption, field_mT)
        csv_path = os.path.join(tmpdir, f"{name}.csv")
        _write_magnettech_csv(csv_path, field_mT, derivative,
                              name=name)
        return csv_path, field_mT, derivative

    def test_is_magnettech_csv_positive(self, tmp_path):
        csv_path, _, _ = self._make_test_csv(str(tmp_path))
        assert _is_magnettech_csv(csv_path)

    def test_is_magnettech_csv_negative(self, tmp_path):
        generic = tmp_path / "generic.csv"
        generic.write_text("a,b,c\n1,2,3\n")
        assert not _is_magnettech_csv(str(generic))

    def test_read_magnettech_csv(self, tmp_path):
        csv_path, field_mT, derivative = (
            self._make_test_csv(str(tmp_path))
        )
        raw = read_magnettech_csv(csv_path)
        np.testing.assert_allclose(
            raw["field_mT"], field_mT, atol=1e-10
        )
        np.testing.assert_allclose(
            raw["intensity"], derivative, atol=1e-10
        )
        assert raw["params"]["Name"] == "test_spectrum"
        assert raw["params"]["Accumulations"] == 10

    def test_load_magnettech_data_gauss(self, tmp_path):
        csv_path, field_mT, _ = (
            self._make_test_csv(str(tmp_path))
        )
        field, intensity, params = load_magnettech_data(
            csv_path, field_unit="G"
        )
        # mT to Gauss conversion
        np.testing.assert_allclose(
            field, field_mT * 10.0, atol=1e-8
        )
        assert params["SOURCE"] == "magnettech"
        assert params["XUNI"] == "G"
        assert params["XPTS"] == 256

    def test_load_magnettech_data_mt(self, tmp_path):
        csv_path, field_mT, _ = (
            self._make_test_csv(str(tmp_path))
        )
        field, _, params = load_magnettech_data(
            csv_path, field_unit="mT"
        )
        np.testing.assert_allclose(
            field, field_mT, atol=1e-10
        )
        assert params["XUNI"] == "mT"

    def test_load_any_epr_csv(self, tmp_path):
        csv_path, _, _ = self._make_test_csv(str(tmp_path))
        field, intensity, params = load_any_epr(csv_path)
        assert params["SOURCE"] == "magnettech"
        assert len(field) == 256

    def test_pipeline_on_magnettech(self, tmp_path):
        """Full pipeline works on Magnettech data."""
        csv_path, _, _ = self._make_test_csv(str(tmp_path))
        field, intensity, params = load_any_epr(csv_path)
        result = process_spectrum(
            field, intensity, n_pts=5
        )
        assert result["converged"]
        assert result["double_integral"] > 0

    def test_find_all_epr_files_csv(self, tmp_path):
        """Directory scanner discovers Magnettech CSVs."""
        self._make_test_csv(str(tmp_path), name="sample1")
        self._make_test_csv(str(tmp_path), name="sample2")
        self._make_test_csv(
            str(tmp_path), name="background_empty"
        )
        files = find_all_epr_files(str(tmp_path))
        names = [name for _, name in files]
        assert "sample1" in names
        assert "sample2" in names
        assert "background_empty" not in names

    def test_find_background_file_csv(self, tmp_path):
        """Background scanner finds Magnettech CSV backgrounds."""
        self._make_test_csv(str(tmp_path), name="sample1")
        self._make_test_csv(
            str(tmp_path), name="background_run"
        )
        bg = find_background_file(str(tmp_path))
        assert bg is not None
        assert "background_run" in bg

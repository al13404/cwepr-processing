"""Basic tests for cwepr_processing."""

import numpy as np
import pytest

from cwepr_processing.processing import process_spectrum


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

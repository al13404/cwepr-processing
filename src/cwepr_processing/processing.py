"""
CW-EPR spectral processing pipeline.

Optimizer notes
---------------
L-BFGS-B is used because it supports parameter bounds (bg_scale
is bounded between 0.5 and 2.0). It is efficient for smooth,
low-dimensional problems like this (only 2 parameters).

ftol is set to 1e-12 (relative change in cost between iterations).
The cost function can reach very small values (1e-9 to 1e-11), so
a tight tolerance ensures the optimizer doesn't stop prematurely.
In practice the gradient tolerance or maxiter usually triggers
termination first.
"""

import numpy as np
from scipy.optimize import minimize


def _correction_pipeline(raw_derivative, field, background_raw,
                         bg_scale, y_center_val, n_pts):
    """Core: bg subtract, quadratic baseline, detrend, integrate."""
    after_bg = raw_derivative - bg_scale * background_raw

    x_start = np.mean(field[:n_pts])
    y_start = np.mean(after_bg[:n_pts])
    x_end = np.mean(field[-n_pts:])
    y_end = np.mean(after_bg[-n_pts:])
    x_center = np.mean(field)

    coeffs = np.polyfit(
        [x_start, x_center, x_end],
        [y_start, y_center_val, y_end], 2
    )
    baseline_quad = np.polyval(coeffs, field)
    detrended = after_bg - baseline_quad

    dx = np.diff(field)
    avg_y = 0.5 * (detrended[:-1] + detrended[1:])
    absorption = np.zeros(len(detrended))
    absorption[1:] = np.cumsum(avg_y * dx)

    return after_bg, baseline_quad, detrended, absorption


def _cost_function(params_opt, raw_derivative, field,
                   background_raw, has_background, n_pts):
    """Cost = edge_start**2 + edge_end**2."""
    if has_background:
        bg_scale, y_center_val = params_opt
    else:
        bg_scale = 0.0
        y_center_val = params_opt[0]

    _, _, _, absorption = _correction_pipeline(
        raw_derivative, field, background_raw,
        bg_scale, y_center_val, n_pts
    )

    edge_start = np.mean(absorption[:n_pts])
    edge_end = np.mean(absorption[-n_pts:])
    return edge_start**2 + edge_end**2


def _linear_absorption_pass(derivative_in, field, n_pts):
    """Residual linear correction in the absorption domain."""
    dx = np.diff(field)
    x_arr = np.arange(len(derivative_in))

    s_avg = np.mean(derivative_in[:n_pts])
    e_avg = np.mean(derivative_in[-n_pts:])
    slope_dt = (e_avg - s_avg) / (len(derivative_in) - 1)
    detrend_line = s_avg + slope_dt * x_arr
    derivative_detrended = derivative_in - detrend_line

    avg_y = 0.5 * (derivative_detrended[:-1]
                    + derivative_detrended[1:])
    absorption = np.zeros(len(derivative_detrended))
    absorption[1:] = np.cumsum(avg_y * dx)

    sa = np.mean(absorption[:n_pts])
    ea = np.mean(absorption[-n_pts:])
    slope_bl = (ea - sa) / (len(absorption) - 1)
    residual_bl = sa + slope_bl * x_arr
    absorption_corrected = absorption - residual_bl

    absorption_offset = (absorption_corrected
                         - np.min(absorption_corrected))

    corrected = np.zeros_like(absorption_corrected)
    corrected[1:-1] = (
        (absorption_corrected[2:] - absorption_corrected[:-2])
        / (field[2:] - field[:-2])
    )
    corrected[0] = (
        (absorption_corrected[1] - absorption_corrected[0])
        / (field[1] - field[0])
    )
    corrected[-1] = (
        (absorption_corrected[-1] - absorption_corrected[-2])
        / (field[-1] - field[-2])
    )

    return {
        "detrend_line": detrend_line,
        "derivative_detrended": derivative_detrended,
        "absorption_raw": absorption,
        "residual_baseline": residual_bl,
        "absorption_corrected": absorption_corrected,
        "absorption_offset": absorption_offset,
        "corrected_derivative": corrected,
    }


def process_spectrum(field, raw_derivative,
                     background_raw=None, n_pts=10):
    """Full CW-EPR processing pipeline.

    Parameters
    ----------
    field : array -- magnetic field (G)
    raw_derivative : array -- raw first-derivative spectrum
    background_raw : array or None -- raw background trace
    n_pts : int -- edge points for anchors/cost (default 10)

    Returns dict with all intermediates and final outputs.
    """
    has_bg = background_raw is not None
    if not has_bg:
        background_raw = np.zeros_like(field)

    if has_bg:
        temp = raw_derivative - background_raw
        y_center_init = 0.5 * (np.mean(temp[:n_pts])
                                + np.mean(temp[-n_pts:]))
        x0 = [1.0, y_center_init]
        bounds = [(0.5, 2.0), (None, None)]
    else:
        y_center_init = 0.5 * (
            np.mean(raw_derivative[:n_pts])
            + np.mean(raw_derivative[-n_pts:])
        )
        x0 = [y_center_init]
        bounds = [(None, None)]

    result = minimize(
        _cost_function, x0=x0,
        method="L-BFGS-B", bounds=bounds,
        args=(raw_derivative, field, background_raw,
              has_bg, n_pts),
        options={"ftol": 1e-12, "maxiter": 10000},
    )

    if has_bg:
        bg_scale_opt, y_center_opt = result.x
    else:
        bg_scale_opt = 0.0
        y_center_opt = result.x[0]

    after_bg, baseline_quad, detrended, absorption_opt = (
        _correction_pipeline(raw_derivative, field,
                             background_raw, bg_scale_opt,
                             y_center_opt, n_pts)
    )

    rc = _linear_absorption_pass(detrended, field, n_pts)

    dx = np.diff(field)
    avg_abs = 0.5 * (rc["absorption_offset"][:-1]
                      + rc["absorption_offset"][1:])
    di_cum = np.zeros(len(field))
    di_cum[1:] = np.cumsum(avg_abs * dx)
    di_total = di_cum[-1]

    normalized = (
        rc["corrected_derivative"] / di_total
        if abs(di_total) > 0
        else rc["corrected_derivative"]
    )

    return {
        "after_bg": after_bg,
        "baseline_quad": baseline_quad,
        "detrended": detrended,
        "absorption_opt": absorption_opt,
        "residual_correction": rc,
        "di_cumulative": di_cum,
        "double_integral": di_total,
        "normalized": normalized,
        "bg_scale": bg_scale_opt,
        "y_center": y_center_opt,
        "cost": result.fun,
        "converged": result.success,
    }

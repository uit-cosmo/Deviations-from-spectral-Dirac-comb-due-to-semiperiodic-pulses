from plot_lorentz_time_series import (
    # generate_fpp,
    calculate_duration_time,
    calculate_fitrange,
    skewed_lorentz,
)
import numpy as np
from scipy.optimize import minimize
from scipy.signal import find_peaks, fftconvolve


def generate_fpp(var, normalized_data, tkern, dt, td, T):
    """generated normalized filtered point process as a fit for given data"""
    pos_peak_loc = find_peaks(normalized_data, height=0.0001)[0]
    neg_peak_loc = find_peaks(-normalized_data, height=0.0001)[0]
    pos_scale, neg_scale, lam, offset = var
    forcing = np.zeros(T.size)
    forcing[pos_peak_loc] = normalized_data[pos_peak_loc] * pos_scale
    forcing[neg_peak_loc] = normalized_data[neg_peak_loc] * neg_scale

    kern = skewed_lorentz(tkern, dt, lam, td, m=offset)

    time_series_fit = fftconvolve(forcing, kern, "same")
    time_series_fit = (time_series_fit - time_series_fit.mean()) / time_series_fit.std()
    return time_series_fit, forcing


def return_peaks(data, T):
    pos_peak_loc = find_peaks(data, height=0.0001)[0]
    neg_peak_loc = find_peaks(-data, height=0.0001)[0]
    peaks = sorted(np.append(pos_peak_loc, neg_peak_loc))
    return T[peaks], data[peaks], peaks


def create_fit_RB(regime, f, dt, PSD, normalized_data, T):
    """calculates fit for Lorenz system time series"""
    symbols = ""

    fitrange = {"4e5": (f < 1600) & (f > 700), "2e6": (f < 5200) & (f > 3500)}
    duration_time = calculate_duration_time(
        f[fitrange[regime]], PSD[(fitrange[regime])]
    )

    kernrad = 2**18
    time_kern = np.arange(-kernrad, kernrad + 1) * dt

    def obj_fun(x):
        return 0.5 * np.sum(
            (
                generate_fpp(x, normalized_data, time_kern, dt, duration_time, T)[0]
                ** 2
                - normalized_data**2
            )
            ** 2
        )

    res = minimize(
        obj_fun,
        [1.0, 1.0, 0.0, 0.0],
        bounds=((0.0, 2.0), (0.0, 2.0), (-0.99, 0.99), (-0.5, 0.5)),
    )
    time_series_fit, forcing = generate_fpp(
        res.x, normalized_data, time_kern, dt, duration_time, T
    )
    return time_series_fit, symbols, duration_time, forcing


def constrained_fit_RB(regime, f, dt, PSD, normalized_data, T):
    """calculates fit for Lorenz system time series"""

    time_peaks, peaks, peak_indices = return_peaks(normalized_data, T)

    symbols = ""

    fitrange = {"4e5": (f < 1600) & (f > 700), "2e6": (f < 5200) & (f > 3500)}
    duration_time = calculate_duration_time(
        f[fitrange[regime]], PSD[(fitrange[regime])]
    )

    kernrad = 2**18
    time_kern = np.arange(-kernrad, kernrad + 1) * dt

    def constraint_func(x):
        fit = generate_fpp(x, normalized_data, time_kern, dt, duration_time, T)[0]
        # print(fit[peak_indices])
        # return fit[peak_indices] - peaks
        return fit[peak_indices[:2]] - peaks[:2]

    def obj_fun(x):
        return 0.5 * np.sum(
            (
                generate_fpp(x, normalized_data, time_kern, dt, duration_time, T)[0]
                ** 2
                - normalized_data**2
            )
            ** 2
        )

    res = minimize(
        obj_fun,
        [1.0, 1.0, 0.0, 0.0],
        bounds=((0.0, 2.0), (0.0, 2.0), (-0.99, 0.99), (-0.5, 0.5)),
        constraints={"type": "eq", "fun": constraint_func},
    )
    time_series_fit, forcing = generate_fpp(
        res.x, normalized_data, time_kern, dt, duration_time, T
    )
    return time_series_fit, symbols, duration_time, forcing

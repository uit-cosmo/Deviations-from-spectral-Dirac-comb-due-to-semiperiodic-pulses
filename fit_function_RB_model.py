from plot_lorentz_time_series import (
    generate_fpp,
    calculate_duration_time,
    calculate_fitrange,
)
import numpy as np
from scipy.optimize import minimize


def create_fit_RB(regime ,f, dt, PSD, normalized_data, T):
    """calculates fit for Lorenz system time series"""
    symbols = ""

    fitrange = {'4e5': (f < 1600) & (f > 700), '2e6': (f < 5200) & (f > 3500)}
    duration_time = calculate_duration_time(
        f[fitrange[regime]], PSD[(fitrange[regime])]
    )

    kernrad = 2**18
    time_kern = np.arange(-kernrad, kernrad + 1) * dt

    def obj_fun(x):
        return 0.5 * np.sum(
            (
                generate_fpp(x, normalized_data, time_kern, dt, duration_time, T) ** 2
                - normalized_data**2
            )
            ** 2
        )

    res = minimize(
        obj_fun,
        [1.0, 1.0, 0.0, 0.0],
        bounds=((0.0, 2.0), (0.0, 2.0), (-0.99, 0.99), (-0.5, 0.5)),
    )
    time_series_fit = generate_fpp(
        res.x, normalized_data, time_kern, dt, duration_time, T
    )
    return time_series_fit, symbols, duration_time

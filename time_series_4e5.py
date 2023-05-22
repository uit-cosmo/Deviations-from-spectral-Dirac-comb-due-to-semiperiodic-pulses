import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import find_peaks, welch, fftconvolve
from scipy.optimize import minimize
import click
import cosmoplots
from plot_lorentz_time_series import (
    generate_fpp,
    calculate_duration_time,
    calculate_fitrange,
)


def create_fit_4e5(f, dt, PSD, normalized_data, T):
    """calculates fit for Lorenz system time series"""
    symbols = ""
    duration_time = calculate_duration_time(
        f[(f < 1600) & (f > 700)], PSD[(f < 1600) & (f > 700)]
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


ts = np.load("RB_time_series_4e5.npy")

intervals_start = 6800

ts_interval = ts[6800:8400]
ts_interval = (ts_interval - np.mean(ts_interval)) / np.std(ts_interval)

dt = 0.01 / 200
f, Pxx = signal.welch(ts_interval, 1 / dt, nperseg=len(ts_interval) / 1)
time = np.linspace(0, dt * len(ts_interval) - dt, num=len(ts_interval))

time_series_fit, symbols, duration_time = create_fit_4e5(
    f, dt, Pxx, ts_interval, time
)

plt.plot(time, ts_interval)
plt.plot(time, time_series_fit)
plt.show()

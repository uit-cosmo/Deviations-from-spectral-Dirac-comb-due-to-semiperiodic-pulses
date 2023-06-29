import numpy as np
from scipy.optimize import minimize
from scipy.signal import find_peaks, fftconvolve


def generate_fpp_K(
    td, normalized_data, tkern, T, lam=0.5, distance=200
):
    """generated normalized filtered point process as a fit for given data"""
    pos_peak_loc = find_peaks(normalized_data, height=1.0, distance=distance)[0]
    forcing = np.zeros(T.size)
    forcing[pos_peak_loc] = normalized_data[pos_peak_loc] * 1
    
    def double_exp(tkern, lam, td):
        kern = np.zeros(tkern.size)
        kern[tkern < 0] = np.exp(tkern[tkern < 0] / lam / td)
        kern[tkern >= 0] = np.exp(-tkern[tkern >= 0] / (1 - lam) / td)
        return kern

    kern = double_exp(tkern, lam, td)
    
    time_series_fit = fftconvolve(forcing, kern, "same")
    time_series_fit = (time_series_fit - time_series_fit.mean()) / time_series_fit.std()
    return time_series_fit, forcing


def create_fit_K(
    f, dt, normalized_data, T, td, lam=0.5, distance=200
):
    """calculates fit for K time series"""
    symbols = ""

    kernrad = 2**18
    time_kern = np.arange(-kernrad, kernrad + 1) * dt

    time_series_fit, forcing = generate_fpp_K(
        td, normalized_data, time_kern, T, distance=distance, lam=lam
    )
    return time_series_fit, symbols, td, forcing

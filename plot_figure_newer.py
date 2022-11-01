import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks
import cosmoplots
from plot_lorentz_time_series import (
    load_system,
    calculate_duration_time,
    calculate_fitrange,
)
from scipy.optimize import minimize, curve_fit
from fppanalysis import get_hist
import scipy as sp


def create_fit(regime, f, dt, PSD, normalized_data, T, constant_amplitudes):
    """calculates fit for Lorenz system time series"""
    fitrange, symbols = calculate_fitrange(regime, f, dt, PSD)

    duration_time = calculate_duration_time(f[fitrange], PSD[fitrange])

    kernrad = 2**18
    time_kern = np.arange(-kernrad, kernrad + 1) * dt

    def obj_fun(x):
        return 0.5 * np.sum(
            (
                generate_forcing(x, normalized_data, T, constant_amplitudes) ** 2
                - normalized_data**2
            )
            ** 2
        )

    res = minimize(
        obj_fun,
        [1.0, 1.0, 0.0, 0.0],
        bounds=((0.0, 2.0), (0.0, 2.0), (-0.99, 0.99), (-0.5, 0.5)),
    )
    time_series_fit = generate_forcing(res.x, normalized_data, T, constant_amplitudes)
    return time_series_fit, fitrange, symbols


def generate_forcing(var, normalized_data, T, constant_amplitudes):
    """generated forcing of normalized filtered point process as a fit for given data"""
    pos_peak_loc = find_peaks(normalized_data, height=1)[0]
    neg_peak_loc = find_peaks(-normalized_data, height=1)[0]
    pos_scale, neg_scale, _, _ = var
    forcing = np.zeros(T.size)
    forcing[pos_peak_loc] = normalized_data[pos_peak_loc] * pos_scale
    forcing[neg_peak_loc] = normalized_data[neg_peak_loc] * neg_scale

    if constant_amplitudes:
        forcing[forcing != 0] = 1

    return forcing


def gamma_wrapper(bins, a, scale, loc):
    return sp.stats.gamma.pdf(bins, a=a, scale=scale, loc=loc)


def lognormal_wrapper(bins, s, scale, loc):
    return sp.stats.lognorm.pdf(bins, s=s, scale=scale, loc=0.6)


def extract_waiting_times(T, forcing):
    arrival_times = T[forcing != 0]
    return np.diff(arrival_times)


def create_figures(fit=True):
    """Creates figure 1 and 5 in manuscript arXiv:2106.15904"""

    regimes = ["rho=28"]

    _ = cosmoplots.set_rcparams_dynamo(plt.rcParams, num_cols=1, ls="thin")

    for regime in regimes:
        T, res = load_system(regime)
        # Choose x-variable for Lorenz
        data = res[0, :]
        normalizes_time_series = (data - data.mean()) / data.std()

        dt = 1e-3
        f, PSD = welch(normalizes_time_series, 1.0 / dt, nperseg=2**19)

        time_series_fit, _, _ = create_fit(
            regime, f, dt, PSD, normalizes_time_series, T, constant_amplitudes=False
        )
        waiting_times = extract_waiting_times(T, time_series_fit)
        bin_centers, waiting_times_hist = get_hist(waiting_times, 64)
        # param_gamma, _ = curve_fit(gamma_wrapper, bin_centers, waiting_times_hist)
        param_lognorm, _ = curve_fit(lognormal_wrapper, bin_centers, waiting_times_hist)
        # print(param_gamma)
        print(param_lognorm)
        plt.plot(bin_centers, waiting_times_hist)
        # plt.plot(
        #     bin_centers,
        #     gamma_wrapper(bin_centers, *param_gamma),
        #     label="gamma dist",
        # )
        plt.plot(
            bin_centers,
            lognormal_wrapper(bin_centers, *param_lognorm),
            label="lognorm dist shape=0.74, scale=0.13, loc=0.6",
        )
        plt.xlabel(r"$\tau_w$")
        plt.ylabel(r"$P(\tau_w)$")
        plt.title(regimes)
        plt.legend()
        plt.show()

        normalized_forcing_fit = (
            time_series_fit - time_series_fit.mean()
        ) / time_series_fit.std()

        f_fit, PSD_fit = welch(normalized_forcing_fit, 1.0 / dt, nperseg=2**19)

        plt.semilogy(f_fit, PSD_fit, c="tab:blue")

        time_series_fit, _, _ = create_fit(
            regime, f, dt, PSD, normalizes_time_series, T, constant_amplitudes=True
        )
        normalized_forcing_fit = (
            time_series_fit - time_series_fit.mean()
        ) / time_series_fit.std()
        f_fit, PSD_fit = welch(normalized_forcing_fit, 1.0 / dt, nperseg=2**19)
        plt.semilogy(
            f_fit,
            PSD_fit,
            c="tab:orange",
            label=r"constant $A$",
        )

        plt.xlabel(r"$f$")
        plt.ylabel(r"$S_{\widetilde{x}}(f)$")
        plt.xlim(0, 20)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    create_figures()

"""This module creates figure 1 and 5 in manuscript arXiv:2106.15904"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, welch, fftconvolve
from scipy.optimize import minimize
import click
import cosmoplots


def load_system(regime):
    """loads in Lorentz system time series created with create_time_series.py"""
    dname = f"lorenz_{regime}.npz"
    F = np.load(dname)
    return F["tb"], F["res"]


def skewed_lorentz(time_kern, dt, lam, td, m=0):
    """Returns skewed lorentzian pulse"""
    f = np.fft.rfftfreq(time_kern.size + 1, dt)
    F = 2 * np.pi * f * td

    def tmp(x):
        res = np.zeros(x.size)
        res[x > 0] = x[x > 0] * np.log(x[x > 0])
        return res

    freq_kern = np.exp(-1.0j * F * m - np.abs(F) - 2.0j * lam * tmp(F) / np.pi)

    ifkern = np.fft.irfft(freq_kern, time_kern.size)
    ifkern = np.fft.fftshift(ifkern)
    ifkern *= td / np.trapz(np.abs(ifkern), time_kern)
    return ifkern


def generate_fpp(var, normalized_data, tkern, dt, td, T):
    """generated normalized filtered point process as a fit for given data"""
    pos_peak_loc = find_peaks(normalized_data, height=1)[0]
    neg_peak_loc = find_peaks(-normalized_data, height=1)[0]
    pos_scale, neg_scale, lam, offset = var
    forcing = np.zeros(T.size)
    forcing[pos_peak_loc] = normalized_data[pos_peak_loc] * pos_scale
    forcing[neg_peak_loc] = normalized_data[neg_peak_loc] * neg_scale

    kern = skewed_lorentz(tkern, dt, lam, td, m=offset)

    time_series_fit = fftconvolve(forcing, kern, "same")
    time_series_fit = (time_series_fit - time_series_fit.mean()) / time_series_fit.std()
    return time_series_fit


def calculate_fitrange(regime, f, dt, S):
    """calculates frequency range used for fit"""
    if regime == "rho=28":
        fitrange = (f > 2) & (f < 25)
        symbols = ""

    else:
        fitrange = {"rho=220": (f < 70), "rho=350": (f < 70)}
        distance = {"rho=220": int(0.25 / dt), "rho=350": int(1.0 / dt)}
        height = {"rho=220": [1e-11, 1e3], "rho=350": [2e-16, 1e3]}

        fitrange = find_peaks(
            S[fitrange[regime]], distance=distance[regime], height=height[regime]
        )[0]
        symbols = "o"
    return fitrange, symbols


def calculate_duration_time(f, S):
    """estimates duration time from slope of power spectral density"""
    p = np.polyfit(f, np.log(S), 1)
    return -p[0] / (4 * np.pi)


def create_fit(regime, f, dt, S, normalized_data, T):
    """calculates fit for Lorenz system time series"""
    fitrange, symbols = calculate_fitrange(regime, f, dt, S)

    duration_time = calculate_duration_time(f[fitrange], S[fitrange])

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
    return time_series_fit, fitrange, symbols


def plot_time_series(regime, T, time_series, fit, time_series_fit):
    """creates plots of time series"""
    plt.figure(f"{regime} time series x compare skew lorenz")

    plt.xlim(0, 10)
    plt.ylim(-2.2, 2.4)
    plt.ylabel(r"$\widetilde{x}$")

    if regime == "rho=28":
        # x-axis is shifted by 10 for illustrative purposes
        plt.plot(T - 10, time_series, label=r"$\rho = 28$")
        plt.text(x=8.25, y=2.1, s=r"$\rho = 28$")
        if fit:
            plt.plot(T - 10, time_series_fit, "--")
        plt.xlabel(r"$t \, \texttt{+} \, 10$")

    if regime == "rho=220":
        plt.plot(T, time_series, label=r"$\rho = 220$")
        plt.text(x=8.25, y=2.1, s=r"$\rho = 220$")
        if fit:
            plt.plot(T, time_series_fit, "--")
        plt.xlabel(r"$t$")

    if regime == "rho=350":
        plt.plot(T, time_series, label=r"$\rho = 350$")
        plt.text(x=8.25, y=2.1, s=r"$\rho = 350$")
        if fit:
            plt.plot(T, time_series_fit, "--")
        plt.xlabel(r"$t$")

    if fit:
        plt.savefig(f"time_series_{regime}_fit.eps", bbox_inches="tight")
    else:
        plt.savefig(f"time_series_{regime}.eps", bbox_inches="tight")


def plot_spectral_density(regime, f, S, fit, f_fit, S_fit, symbols, fitrange):
    """creates plots of power spectral density"""
    plt.figure(f"{regime} PSD x fit td")

    plt.semilogy(f, S, c="tab:blue")

    if fit:
        plt.semilogy(f[fitrange], S[fitrange], symbols, c="tab:blue")
        plt.semilogy(f_fit[fitrange], S_fit[fitrange], symbols, c="tab:orange")
        plt.semilogy(f_fit, S_fit, c="tab:orange")

    plt.xlim(-2, 40)
    plt.ylim(1e-23, 1e4)
    plt.xlabel(r"$f$")
    plt.ylabel(r"$S_{\widetilde{x}}(f)$")

    if regime == "rho=28":
        plt.text(x=31.5, y=2, s=r"$\rho = 28$")
    if regime == "rho=220":
        plt.text(x=31.5, y=2, s=r"$\rho = 220$")
    if regime == "rho=350":
        plt.text(x=31.5, y=2, s=r"$\rho = 350$")

    if fit:
        plt.savefig(f"PSD_{regime}_fit.eps", bbox_inches="tight")
    else:
        plt.savefig(f"PSD_{regime}.eps", bbox_inches="tight")


@click.command()
@click.option("--fit/--no-fit", default=False, help="Fit function in figure 1&5 ")
def create_figures(fit):
    """Creates figure 1 and 5 in manuscript arXiv:2106.15904"""

    regimes = ["rho=28", "rho=220", "rho=350"]

    axes_size = cosmoplots.set_rcparams_dynamo(plt.rcParams, num_cols=1, ls="thin")

    for regime in regimes:
        T, res = load_system(regime)
        # Choose x-variable for Lorenz
        data = res[0, :]
        normalizes_time_series = (data - data.mean()) / data.std()

        dt = 1e-3
        f, S = welch(normalizes_time_series, 1.0 / dt, nperseg=2**20)

        time_series_fit, fitrange, symbols = create_fit(
            regime, f, dt, S, normalizes_time_series, T
        )
        f_fit, S_fit = welch(time_series_fit, 1.0 / dt, nperseg=2**20)

        plot_spectral_density(regime, f, S, fit, f_fit, S_fit, symbols, fitrange)
        plot_time_series(regime, T, normalizes_time_series, fit, time_series_fit)


if __name__ == "__main__":
    create_figures()

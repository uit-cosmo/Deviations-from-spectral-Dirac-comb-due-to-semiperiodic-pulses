"""
Power spectrum and autocorrelation of a process with periodic arrivals and Gaussian jitter.
"""

import numpy as np
from scipy import signal
import support_functions as sf
import fppanalysis as fa
import superposedpulses.forcing as frc
import superposedpulses.point_model as pm
import superposedpulses.pulse_shape as ps
import matplotlib.pyplot as plt
import cosmoplots

plt.style.use("cosmoplots.default")

fig, ax = cosmoplots.figure_multiple_rows_columns(1, 2)
cosmoplots.change_log_axis_base(ax[0], "y")

Sigma = [0.05, 0.5, 5]
Sigmalab = [
    r"$\langle w \rangle/100$",
    r"$\langle w \rangle/10$",
    r"$\langle w \rangle$",
]
waiting_time = 5
dt = 1e-2

# All time is normalized to td.

model = pm.PointModel(waiting_time=waiting_time, total_duration=100000, dt=dt)
model.set_pulse_shape(ps.LorentzShortPulseGenerator(tolerance=1e-5))

for i, sigma in enumerate(Sigma):
    model.set_custom_forcing_generator(
        sf.ForcingQuasiPeriodicAsymLapAmp(sigma=sigma, beta=0.0)
    )

    T, S = model.make_realization()
    S_norm = S - S.mean()

    f, Pxx = signal.welch(x=S_norm, fs=1.0 / dt, nperseg=S.size / 30)
    ax[0].plot(f, Pxx, label=r"$\sigma = $" + Sigmalab[i], c="C{}".format(i))

    if i < 2:
        fitrange = signal.find_peaks(
            Pxx[(f < 1)], distance=int((dt / waiting_time) ** (-1)), height=[5e-4, 1e3]
        )[0]
        ax[0].plot(f[fitrange][1:], Pxx[fitrange][1:], "o", c="C{}".format(i))

    tb, R = fa.corr_fun(S_norm, S_norm, dt=dt, norm=False, biased=True, method="auto")

    # divide by max to show normalized Phi
    ax[1].plot(
        tb[abs(tb) < 50],
        R[abs(tb) < 50] / np.max(R),
        label=r"$\sigma = $" + Sigmalab[i],
        c="C{}".format(i),
    )


def Lorentz_PSD(theta):
    """PSD of a single Lorentz pulse with duration time td = 1"""
    return 2 * np.pi * np.exp(-2 * np.abs(theta))


def find_nearest(array, value):
    """returns array of peaks in PSD"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def spectra_analytical(omega, gamma, A_rms, A_mean, sigma, T):
    I_2 = 1 / (2 * np.pi)
    first_term = (
        gamma
        * (A_rms**2 + A_mean**2 * (1 - np.exp(-((sigma * omega) ** 2))))
        * I_2
        * Lorentz_PSD(omega)
    )

    tmp = np.zeros(omega.size)
    for n in range(-1000, 1000):
        index = 2 * np.pi * n * gamma
        tmp = np.where(np.abs(omega - find_nearest(omega, index)) > 0.001, tmp, 1)

    second_term = (
        2
        * np.pi
        * gamma**2
        * A_mean**2
        * I_2
        * Lorentz_PSD(omega)
        * np.exp(-((sigma * omega) ** 2))
    ) * tmp
    return 2 * (first_term + second_term * T)


window_size = 1e5 / (2 * np.pi * 30)
for label, ls, symb, sigma in zip(
    [r"$S_{{\Phi}}(\tau_\mathrm{d} f)$", None, None],
    ["--", "-.", ":"],
    ["o", "d", "*"],
    Sigma,
):
    PSD = spectra_analytical(
        2 * np.pi * f,
        gamma=1 / waiting_time,
        A_rms=1,
        A_mean=1,
        sigma=sigma,
        T=window_size,
    )
    ax[0].plot(f, PSD, ls + "k", label=label)
    # fpeaks = gamma*np.arange(1,10)
    # ax[0].scatter(fpeaks, )


def Lorentz_AC_basic(t):
    return 4 / (4 + t**2)


tb = np.linspace(0, 50, 1000)
ax[1].plot(tb, Lorentz_AC_basic(tb), ":k", label=r"$\rho_\phi(t/\tau_\mathrm{d})$")

ax[0].set_xlabel(r"$\tau_\mathrm{d} f$")
ax[0].set_ylabel(r"$S_{{\Phi}}(\tau_\mathrm{d} f)$")
ax[0].set_xlim(-0.03, 0.8)
ax[0].set_ylim(1e-4, 1e2)
ax[0].legend()

ax[1].set_xlim(0, 50)
ax[1].set_xlabel(r"$t/\tau_\mathrm{d}$")
ax[1].set_ylabel(r"$R_{\widetilde{\Phi}}(t/\tau_\mathrm{d})$")
ax[1].legend()

fig.savefig("gaussian_jitter_wait.eps")

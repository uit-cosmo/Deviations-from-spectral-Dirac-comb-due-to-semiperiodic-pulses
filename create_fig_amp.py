"""
Power spectral density and autocorrelation function
of asymetrically laplace distributed amplitudes with different asymmetry parameters.

Creates fig. 4 in the most recent manuscript.
"""

import numpy as np
from scipy import signal
import fppanalysis as fa
import support_functions as sf
import superposedpulses.point_model as pm
import superposedpulses.pulse_shape as ps
from scipy.signal import find_peaks
from closedexpressions import PSD_periodic_arrivals, autocorr_periodic_arrivals

import matplotlib.pyplot as plt
import cosmoplots

plt.style.use("cosmoplots.default")

total_duration = 100_000
segments = 30  # Segments in Welch
dt = 1e-2


def height(f):
    return 50 * np.exp(-2 * 2 * np.pi * f)


fig, ax = cosmoplots.figure_multiple_rows_columns(1, 2)
cosmoplots.change_log_axis_base(ax[0], "y")

model = pm.PointModel(waiting_time=5.0, total_duration=total_duration, dt=dt)
model.set_pulse_shape(ps.LorentzShortPulseGenerator(tolerance=1e-5))


for ind, control_parameter, label in zip(
    [0, 1, 2], [0.0, 0.4, 0.5], [r"$0$", r"$2/5$", r"$1/2$"]
):
    model.set_custom_forcing_generator(
        sf.ForcingQuasiPeriodicAsymLapAmp(sigma=0.0, beta=control_parameter)
    )

    T, S = model.make_realization()
    forcing = model.get_last_used_forcing()

    S_norm = (S - S.mean()) / S.std()

    f, Pxx = signal.welch(
        x=S_norm, fs=1 / dt, nperseg=int(total_duration / dt / segments)
    )
    ax[0].plot(f, Pxx)  # , label=r"$\lambda = $" + label, c="C{}".format(ind))

    fitrange = find_peaks(
        Pxx[(f < 1)], distance=int(0.1 / dt), height=height(f[f < 1])
    )[0]
    ax[0].plot(f[fitrange], Pxx[fitrange], "o", c="C{}".format(ind))

    tb, R = fa.corr_fun(S_norm, S_norm, dt=0.01, norm=False, biased=True, method="auto")
    ax[1].plot(tb, R, label=r"$\lambda = $" + label, c="C{}".format(ind))


window_size_angular = S.size * 1e-2 / (2 * np.pi)
for A_mean, label, ls in zip([1, 0], [r"$0$", r"$1/2$"], ["--k", ":C7"]):
    PSD = PSD_periodic_arrivals(
        2 * np.pi * f,
        td=1,
        gamma=0.2,
        A_rms=1,
        A_mean=A_mean,
        T=window_size_angular / segments,
    )
    ax[0].plot(
        f,
        PSD,
        ls,
        # label=r"$",
    )
    good = (PSD > height(f)) & (f < 0.9)
    ax[0].plot(f[good], PSD[good], "kx")

    t = np.linspace(0, 50, 1000)

    R_an = autocorr_periodic_arrivals(t, gamma=0.2, A_mean=A_mean, A_rms=1, norm=True)
    ax[1].plot(
        t,
        R_an,
        ls,
        label=r"$\lambda = $" + label + r"$\,\mathrm{an.}$",
    )

# ax[0].legend()
ax[0].set_xlim(-0.03, 1)
ax[0].set_ylim(1e-4, 1e3)
ax[0].set_xlabel(r"$\tau_\mathrm{d} f$")
ax[0].set_ylabel(r"$S_{\widetilde{\Phi}}(\tau_\mathrm{d} f)$")

ax[1].set_xlim(0, 50)
ax[1].set_xlabel(r"$t/\tau_\mathrm{d}$")
ax[1].set_ylabel(r"$R_{\widetilde{\Phi}}(t/\tau_\mathrm{d})$")
ax[1].legend()

fig.savefig("asymlaplaceamp.eps")

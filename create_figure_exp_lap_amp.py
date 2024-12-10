"""
Power spectral density and autocorrelation function for periodic arrivals and exponentially and laplace distributed amplitudes.

Figure 3 in the paper.

TODO: Fix dirac delta amplitudes.
"""

import numpy as np
from scipy import signal
import fppanalysis as fa
import support_functions as sf
import superposedpulses.point_model as pm
import superposedpulses.pulse_shape as ps
from closedexpressions import PSD_periodic_arrivals, autocorr_periodic_arrivals
import matplotlib.pyplot as plt
import cosmoplots

plt.style.use("cosmoplots.default")

fig, ax = cosmoplots.figure_multiple_rows_columns(rows=1, columns=2)
cosmoplots.change_log_axis_base(ax[0], "y")

model = pm.PointModel(waiting_time=5, total_duration=100_000, dt=0.01)
model.set_pulse_shape(ps.LorentzShortPulseGenerator(tolerance=1e-5))
model.set_custom_forcing_generator(
    sf.ForcingQuasiPeriodicAsymLapAmp(sigma=0.0, beta=0.0)
)

T, S = model.make_realization()
print(T[-1])

S_norm = (S - S.mean()) / S.std()

f, Pxx = signal.welch(x=S_norm, fs=100, nperseg=S.size / 10)
ax[0].plot(f, Pxx, "C0")  # , label=r"$A \sim \mathrm{Exp}$")

limit = 50 * np.exp(-4 * np.pi * f)
good = (Pxx > limit) & (f < 0.9)
ax[0].plot(f[good], Pxx[good], "o", c="C0", markersize=3)


window_size_angular = S.size * 1e-2 / (2 * np.pi * 10)
#
PSD = PSD_periodic_arrivals(
    2 * np.pi * f, td=1, gamma=0.2, A_rms=1, A_mean=1, T=window_size_angular
)
ax[0].plot(
    f,
    PSD,
    "--k",
    # label=r"$S_{\widetilde{\Phi}}(\tau_\mathrm{d} f), \, \langle A \rangle \ne 0$",
)
good = (PSD > limit) & (f < 0.9)
ax[0].plot(f[good], PSD[good], "k*")

tb, R = fa.corr_fun(S_norm, S_norm, dt=0.01, norm=False, biased=True, method="auto")
ax[1].plot(tb, R, label=r"$A \sim \mathrm{Exp}$")

t = np.linspace(0, 50, 1000)
R_an = autocorr_periodic_arrivals(t, gamma=0.2, A_mean=1, A_rms=1, norm=True)
ax[1].plot(
    t,
    R_an,
    "--k",
    label=r"$\langle A \rangle \ne 0$",
)

model = pm.PointModel(waiting_time=5, total_duration=100000, dt=0.01)
model.set_pulse_shape(ps.LorentzShortPulseGenerator(tolerance=1e-5))
model.set_custom_forcing_generator(
    sf.ForcingQuasiPeriodicAsymLapAmp(sigma=0.0, beta=0.5)
)

T, S = model.make_realization()

S_norm = (S - S.mean()) / S.std()

f, Pxx = signal.welch(x=S_norm, fs=100, nperseg=S.size / 10)
ax[0].plot(f, Pxx, "C1")  # , label=r"$A \sim \mathrm{Laplace}$")

PSD = PSD_periodic_arrivals(2 * np.pi * f, td=1, gamma=0.2, A_rms=1, A_mean=0, T=T[-1])
ax[0].plot(
    f,
    PSD,
    "--g",
    # label=r"$S_{\widetilde{\Phi}}(\tau_\mathrm{d} f), \, \langle A \rangle = 0$",
)

tb, R = fa.corr_fun(S_norm, S_norm, dt=0.01, norm=False, biased=True, method="auto")
ax[1].plot(tb, R, label=r"$A \sim \mathrm{Laplace}$")

R_an = autocorr_periodic_arrivals(t, gamma=0.2, A_mean=0, A_rms=1, norm=True)
ax[1].plot(
    t,
    R_an,
    "--g",
    label=r"$\langle A \rangle = 0$",
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

fig.savefig("exp_lap.eps")

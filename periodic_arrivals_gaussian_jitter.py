import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from support_functions import *
import model.forcing as frc
import model.point_model as pm
import model.pulse_shape as ps
import cosmoplots
from closedexpressions import PSD_periodic_arrivals, autocorr_periodic_arrivals


axes_size = cosmoplots.set_rcparams_dynamo(plt.rcParams, num_cols=1, ls="thin")

fig_PSD = plt.figure()
ax1 = fig_PSD.add_axes(axes_size)
fig_AC = plt.figure()
ax2 = fig_AC.add_axes(axes_size)


class ForcingQuasiPeriodic(frc.ForcingGenerator):
    def __init__(self, kappa):
        self.kappa = kappa

    def get_forcing(self, times: np.ndarray, gamma: float) -> frc.Forcing:
        total_pulses = int(max(times) * gamma)
        periodic_waiting_times = np.ones(shape=total_pulses)
        waiting_times_jitter = np.random.uniform(
            low=-self.kappa / 2, high=self.kappa / 2, size=total_pulses
        )
        # * 100 for dt correction
        waiting_times = (periodic_waiting_times + waiting_times_jitter) * 100 / gamma
        arrival_times = np.add.accumulate(waiting_times)
        arrival_time_indx = np.rint(arrival_times).astype(int)
        arrival_time_indx -= arrival_time_indx[0]  # set first pulse to t = 0
        # check whether events are sampled with arrival time > times[-1]
        number_of_overshotings = len(arrival_time_indx[arrival_time_indx > times.size])
        total_pulses -= number_of_overshotings
        arrival_time_indx = arrival_time_indx[arrival_time_indx < times.size]

        amplitudes = np.random.default_rng().exponential(scale=1.0, size=total_pulses)
        durations = np.ones(shape=total_pulses)

        return frc.Forcing(
            total_pulses,
            times[arrival_time_indx],
            amplitudes,
            durations,
        )

    def set_amplitude_distribution(
        self,
        amplitude_distribution_function,
    ):
        pass

    def set_duration_distribution(self, duration_distribution_function):
        pass


model = pm.PointModel(gamma=0.2, total_duration=100000, dt=0.01)
model.set_pulse_shape(ps.LorentzShortPulseGenerator(tolerance=1e-5))

colors = ["tab:blue", "tab:orange", "tab:olive"]
for i, kappa in enumerate([0.1, 0.4, 1.0]):
    model.set_custom_forcing_generator(ForcingQuasiPeriodic(kappa=kappa))

    T, S = model.make_realization()
    forcing = model.get_last_used_forcing()
    amp = forcing.amplitudes

    S_norm = (S - S.mean()) / S.std()

    f, Pxx = signal.welch(x=S_norm, fs=100, nperseg=S.size / 30)
    ax1.semilogy(f, Pxx, label=rf"$\kappa = {kappa}$", color=colors[i])

    tb, R = corr_fun(S_norm, S_norm, dt=0.01, norm=False, biased=True, method="auto")
    ax2.plot(tb, R, label=rf"$\kappa = {kappa}$", color=colors[i])

# PSD = PSD_periodic_arrivals(2 * np.pi * f, td=1, gamma=0.2, A_rms=1, A_mean=1, dt=0.01)
# ax1.semilogy(f, PSD, "--k", label=r"$S_{\widetilde{\Phi}}(\tau_\mathrm{d} f)$")
# t = np.linspace(0, 50, 1000)
# R_an = autocorr_periodic_arrivals(t, 0.2, 1, 1)
# ax2.plot(t, R_an, "--k", label=r"$R_{\widetilde{\Phi}}(t/\tau_\mathrm{d})$")

ax1.set_xlabel(r"$\tau_\mathrm{d} f$")
ax1.set_ylabel(r"$S_{\widetilde{\Phi}}(\tau_\mathrm{d} f)$")
ax1.set_xlim(-0.03, 1)
ax1.set_ylim(1e-4, 1e2)
ax1.legend()

ax2.set_xlim(0, 50)
ax2.set_xlabel(r"$t/\tau_\mathrm{d}$")
ax2.set_ylabel(r"$R_{\widetilde{\Phi}}(t/\tau_\mathrm{d})$")
ax2.legend()
cosmoplots.change_log_axis_base(ax1, "y", base=10)

# fig_PSD.savefig("PSD_different_kappa.eps", bbox_inches="tight")
# fig_AC.savefig("AC_different_kappa.eps", bbox_inches="tight")

plt.show()

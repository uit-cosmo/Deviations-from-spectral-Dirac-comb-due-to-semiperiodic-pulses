import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from support_functions import *
import model.forcing as frc
import model.point_model as pm
import model.pulse_shape as ps
import cosmoplots

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
        waiting_times = (
            np.random.uniform(
                low=1 - self.kappa / 2, high=1 + self.kappa / 2, size=total_pulses
            )
            * 100  # multiplited with inverse dt
        ) / gamma
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
model.set_custom_forcing_generator(ForcingQuasiPeriodic(kappa=0.1))

T, S = model.make_realization()
amp = np.random.default_rng().exponential(scale=1.0, size=19999)

S_norm = (S - S.mean()) / S.std()


f, Pxx = signal.welch(x=S_norm, fs=100, nperseg=S.size / 30)

ax1.semilogy(f, Pxx, label=r"$\kappa = 0.1$")
PSD = PSD_periodic_arrivals(
    2 * np.pi * f, td=1, gamma=0.2, Arms=amp.std(), Am=np.mean(amp), S=S
)

ax1.set_xlim(-0.2, 12)
ax1.set_ylim(1e-14, 1e3)

ax1.set_xlabel(r"$f$")
ax1.set_ylabel(r"$S_{\widetilde{\Phi}}(f)$")

tb, R = corr_fun(S_norm, S_norm, dt=0.01, norm=False, biased=True, method="auto")

ax2.plot(tb, R, label=r"$\kappa = 0.1$")

model.set_custom_forcing_generator(ForcingQuasiPeriodic(kappa=0.4))

T, S = model.make_realization()

S_norm = (S - S.mean()) / S.std()
t, R_an = calculate_R_an(1, 1, 0.2)

f, Pxx = signal.welch(x=S_norm, fs=100, nperseg=S.size / 30)

ax1.semilogy(f, Pxx, label=r"$\kappa = 0.4$")
PSD = PSD_periodic_arrivals(
    2 * np.pi * f, td=1, gamma=0.2, Arms=amp.std(), Am=np.mean(amp), S=S
)

ax1.set_xlim(-0.2, 12)
ax1.set_ylim(1e-14, 1e3)

ax1.set_xlabel(r"$f$")
ax1.set_ylabel(r"$S_{\widetilde{\Phi}}(f)$")

tb, R = corr_fun(S_norm, S_norm, dt=0.01, norm=False, biased=True, method="auto")

ax2.plot(tb, R, label=r"$\kappa = 0.4$")

model.set_custom_forcing_generator(ForcingQuasiPeriodic(kappa=1.0))

T, S = model.make_realization()

S_norm = (S - S.mean()) / S.std()
t, R_an = calculate_R_an(1, 1, 0.2)

f, Pxx = signal.welch(x=S_norm, fs=100, nperseg=S.size / 30)

ax1.semilogy(f, Pxx, label=r"$\kappa = 1$")
PSD = PSD_periodic_arrivals(
    2 * np.pi * f, td=1, gamma=0.2, Arms=amp.std(), Am=np.mean(amp), S=S
)
ax1.semilogy(f, PSD, "--k", label=r"$S_{\widetilde{\Phi}}(f)$")

tb, R = corr_fun(S_norm, S_norm, dt=0.01, norm=False, biased=True, method="auto")

ax2.plot(tb, R, label=r"$\kappa = 1$")

ax2.plot(t, R_an, "--k", label=r"$R_{\widetilde{\Phi}}(t)$")

ax1.legend()
ax1.set_xlim(-0.03, 1)
ax1.set_ylim(1e-4, 1e2)

ax2.set_xlim(0, 50)
ax2.set_xlabel(r"$t$")
ax2.set_ylabel(r"$R_{\widetilde{\Phi}}(t)$")
ax2.legend()
cosmoplots.change_log_axis_base(ax1, "y", base=10)

fig_PSD.savefig("PSD_different_kappa.eps", bbox_inches="tight")
fig_AC.savefig("AC_different_kappa.eps", bbox_inches="tight")

plt.show()

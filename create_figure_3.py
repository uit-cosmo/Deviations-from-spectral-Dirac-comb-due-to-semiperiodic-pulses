import numpy as np
from scipy import signal
import cosmoplots
import matplotlib as mpl
import matplotlib.pyplot as plt
from support_functions import *
import superposedpulses.forcing as frc
import superposedpulses.point_model as pm
import superposedpulses.pulse_shape as ps
from scipy.signal import find_peaks

mpl.style.use("cosmoplots.default")

fig_PSD = plt.figure()
ax1 = fig_PSD.gca()
cosmoplots.change_log_axis_base(ax1, "y")
fig_AC = plt.figure()
ax2 = fig_AC.gca()


class AsymLaplaceAmp(frc.ForcingGenerator):
    def __init__(self, control_parameter):
        self.control_parameter = control_parameter

    def get_forcing(self, times: np.ndarray, gamma: float) -> frc.Forcing:
        total_pulses = int(max(times) * gamma)
        arrival_time_indx = (
            np.arange(start=0, stop=99994, step=5) * 100
        )  # multiplied with inverse dt
        amplitudes = sample_asymm_laplace(
            alpha=0.5
            / np.sqrt(
                1.0 - 2.0 * self.control_parameter * (1.0 - self.control_parameter)
            ),
            kappa=self.control_parameter,
            size=total_pulses,
        )
        durations = np.ones(shape=total_pulses)
        return frc.Forcing(
            total_pulses, times[arrival_time_indx], amplitudes, durations
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

plot_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

for control_parameter, color in zip([0.2, 0.4, 0.45, 0.48], plot_colors):
    model.set_custom_forcing_generator(
        AsymLaplaceAmp(control_parameter=control_parameter)
    )

    T, S = model.make_realization()
    forcing = model.get_last_used_forcing()
    amp = forcing.amplitudes

    S_norm = (S - S.mean()) / S.std()

    f, Pxx = signal.welch(x=S_norm, fs=100, nperseg=S.size / 30)
    ax1.plot(f, Pxx, label=rf"$\lambda = {control_parameter}$", c=color)

    fitrange = find_peaks(Pxx[(f < 1)], distance=500, height=[5e-4, 1e3])[0]
    ax1.plot(f[fitrange][1:], Pxx[fitrange][1:], "o", c=color)

    tb, R = corr_fun(S_norm, S_norm, dt=0.01, norm=False, biased=True, method="auto")
    ax2.plot(tb, R, label=rf"$\lambda = {control_parameter}$", c=color)

ax1.legend()
ax1.set_xlim(-0.03, 1)
ax1.set_ylim(1e-4, 1e3)
ax1.set_xlabel(r"$\tau_\mathrm{d} f$")
ax1.set_ylabel(r"$S_{\widetilde{\Phi}}(\tau_\mathrm{d} f)$")

ax2.set_xlim(0, 50)
ax2.set_xlabel(r"$t/\tau_\mathrm{d}$")
ax2.set_ylabel(r"$R_{\widetilde{\Phi}}(t/\tau_\mathrm{d})$")
ax2.legend()

fig_PSD.savefig("PSD_asym_lap.eps", bbox_inches="tight")
fig_AC.savefig("AC_asym_lap.eps", bbox_inches="tight")

plt.show()

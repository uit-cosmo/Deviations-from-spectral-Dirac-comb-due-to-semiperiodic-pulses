"""
Power spectral density and autocorrelation function
of asymetrically laplace distributed amplitudes with different asymmetry parameters.

Creates fig. 4 in the most recent manuscript.
"""

import numpy as np
from scipy import signal
import fppanalysis as fa
import support_functions as sf
import superposedpulses.forcing as frc
import superposedpulses.point_model as pm
import superposedpulses.pulse_shape as ps
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
import cosmoplots

plt.style.use("cosmoplots.default")

fig, ax = cosmoplots.figure_multiple_rows_columns(1, 2)
cosmoplots.change_log_axis_base(ax[0], "y")


class AsymLaplaceAmp(frc.ForcingGenerator):
    def __init__(self, control_parameter):
        self.control_parameter = control_parameter

    def get_forcing(self, times: np.ndarray, waiting_time: float) -> frc.Forcing:
        total_pulses = int(max(times) / waiting_time)
        arrival_time_indx = (
            np.arange(start=0, stop=99994, step=5) * 100
        )  # multiplied with inverse dt
        amplitudes = sf.sample_asymm_laplace(
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


model = pm.PointModel(waiting_time=5.0, total_duration=100_000, dt=0.01)
model.set_pulse_shape(ps.LorentzShortPulseGenerator(tolerance=1e-5))

for color, control_parameter in enumerate([0.2, 0.4, 0.45, 0.48]):
    model.set_custom_forcing_generator(
        AsymLaplaceAmp(control_parameter=control_parameter)
    )

    T, S = model.make_realization()
    forcing = model.get_last_used_forcing()

    S_norm = (S - S.mean()) / S.std()

    f, Pxx = signal.welch(x=S_norm, fs=100, nperseg=S.size / 30)
    ax[0].plot(f, Pxx, label=rf"$\lambda = {control_parameter}$", c="C{}".format(color))

    height = 20 * np.exp(-2 * 2 * np.pi * f)

    fitrange = find_peaks(Pxx[(f < 1)], distance=10, height=height[f < 1])[0]
    ax[0].plot(f[fitrange], Pxx[fitrange], "o", c="C{}".format(color))

    tb, R = fa.corr_fun(S_norm, S_norm, dt=0.01, norm=False, biased=True, method="auto")
    ax[1].plot(tb, R, label=rf"$\lambda = {control_parameter}$")

ax[0].legend()
ax[0].set_xlim(-0.03, 1)
ax[0].set_ylim(1e-4, 1e3)
ax[0].set_xlabel(r"$\tau_\mathrm{d} f$")
ax[0].set_ylabel(r"$S_{\widetilde{\Phi}}(\tau_\mathrm{d} f)$")

ax[1].set_xlim(0, 50)
ax[1].set_xlabel(r"$t/\tau_\mathrm{d}$")
ax[1].set_ylabel(r"$R_{\widetilde{\Phi}}(t/\tau_\mathrm{d})$")
ax[1].legend()

fig.savefig("asym_laplace_amp.eps")

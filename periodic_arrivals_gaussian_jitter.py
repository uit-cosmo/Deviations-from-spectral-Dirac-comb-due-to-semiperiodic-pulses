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
# fig_AC = plt.figure()
# ax2 = fig_AC.add_axes(axes_size)


class ForcingQuasiPeriodic(frc.ForcingGenerator):
    def __init__(self, variance):
        self.sigma = np.sqrt(variance)

    def get_forcing(self, times: np.ndarray, gamma: float) -> frc.Forcing:
        total_pulses = int(max(times) * gamma)
        periodic_waiting_times = np.arange(1, total_pulses + 1)
        waiting_times_jitter = np.random.normal(
            loc=1, scale=self.sigma, size=total_pulses
        )
        # * 100 for dt correction
        arrival_times = (periodic_waiting_times + waiting_times_jitter) * 100 / gamma

        # arrival_times = np.add.accumulate(waiting_times)
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
for i, variance in enumerate([0.01]):#, 0.01, 0.04]):
    model.set_custom_forcing_generator(ForcingQuasiPeriodic(variance=variance))

    T, S = model.make_realization()
    forcing = model.get_last_used_forcing()
    amp = forcing.amplitudes

    # S_norm = (S - S.mean()) / S.std()
    S_norm = S - S.mean()

    f, Pxx = signal.welch(x=S_norm, fs=100, nperseg=S.size / 30)
    ax1.semilogy(f, Pxx, label=rf"$\sigma = {variance}$", color=colors[i])

    # tb, R = corr_fun(S_norm, S_norm, dt=0.01, norm=False, biased=True, method="auto")
    # ax2.plot(tb, R, label=rf"$\sigma = {variance}$", color=colors[i])

# PSD = PSD_periodic_arrivals(2 * np.pi * f, td=1, gamma=0.2, A_rms=1, A_mean=1, dt=0.01)
# ax1.semilogy(f, PSD, "--k", label=r"$S_{\widetilde{\Phi}}(\tau_\mathrm{d} f)$")
# t = np.linspace(0, 50, 1000)
# R_an = autocorr_periodic_arrivals(t, 0.2, 1, 1)
# ax2.plot(t, R_an, "--k", label=r"$R_{\widetilde{\Phi}}(t/\tau_\mathrm{d})$")


def Lorentz_PSD(theta):
    """PSD of a single Lorentz pulse with duration time td = 1"""
    return 2 * np.pi * np.exp(-2 * np.abs(theta))


def find_nearest(array, value):
    """returns array of parks in PSD"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def spectra_analytical(omega, gamma, A_rms, A_mean, sigma, dt):
    nu = sigma * gamma
    Omega = omega / gamma
    I_2 = 1 / (2 * np.pi)
    first_term = gamma * A_rms**2 * I_2 * Lorentz_PSD(omega)

    tmp = np.zeros(omega.size)
    for n in range(-1000, 1000):
        index = 2 * np.pi * n
        tmp = np.where(np.abs(Omega - find_nearest(Omega, index)) > 0.001, tmp, 1)

    second_term = (
        2
        * np.pi**2
        * gamma**2
        * A_mean**2
        * I_2
        * Lorentz_PSD(omega)
        * np.exp(-(nu**2) * Omega**2)
    ) * tmp
    return 2 * (first_term + second_term / dt)

gamma = 0.2
PSD = spectra_analytical(
    2 * np.pi * f, gamma=gamma, A_rms=1, A_mean=1, sigma=0.1/gamma, dt=0.01
)
ax1.semilogy(f, PSD, "--k", label=r"$S_{{\Phi}}(\tau_\mathrm{d} f)$")

# PSD = spectra_analytical(
#     2 * np.pi * f, gamma=0.2, A_rms=1, A_mean=1, sigma=0.1, dt=0.01
# )

# ax1.semilogy(f, PSD, "--k", label=r"$S_{{\Phi}}(\tau_\mathrm{d} f)$")
ax1.set_xlabel(r"$\tau_\mathrm{d} f$")
ax1.set_ylabel(r"$S_{{\Phi}}(\tau_\mathrm{d} f)$")
ax1.set_xlim(-0.03, 1)
ax1.set_ylim(1e-4, 1e2)
ax1.legend()
#
# ax2.set_xlim(0, 50)
# ax2.set_xlabel(r"$t/\tau_\mathrm{d}$")
# ax2.set_ylabel(r"$R_{\widetilde{\Phi}}(t/\tau_\mathrm{d})$")
# ax2.legend()
# cosmoplots.change_log_axis_base(ax1, "y", base=10)

# fig_PSD.savefig("PSD_gaussian_jitter.eps", bbox_inches="tight")
# fig_AC.savefig("AC_gaussian_jitter.eps", bbox_inches="tight")

plt.show()

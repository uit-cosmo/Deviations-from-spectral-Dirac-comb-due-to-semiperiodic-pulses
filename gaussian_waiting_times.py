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
    def __init__(self, variance):
        self.sigma = np.sqrt(variance)

    def get_forcing(self, times: np.ndarray, gamma: float) -> frc.Forcing:
        total_pulses = int(max(times) * gamma)
        waiting_times = (
            np.random.normal(loc=1, scale=self.sigma, size=total_pulses)
            * 100  # multiplied with inverse dt
        ) / gamma
        print(np.mean(waiting_times))
        print(np.std(waiting_times))
        # plt.hist(waiting_times)
        # plt.show()
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
for i, variance in enumerate([0.0025, 0.01, 1]):  # , 0.4, 3.0]):
    model.set_custom_forcing_generator(ForcingQuasiPeriodic(variance=variance))

    T, S = model.make_realization()
    forcing = model.get_last_used_forcing()
    amp = forcing.amplitudes

    # S_norm = (S - S.mean()) / S.std()
    S_norm = S - S.mean()

    f, Pxx = signal.welch(x=S_norm, fs=100, nperseg=S.size / 30)
    ax1.semilogy(f, Pxx, label=rf"$\sigma= {variance}$", color=colors[i])

    tb, R = corr_fun(S_norm, S_norm, dt=0.01, norm=False, biased=True, method="auto")
    ax2.plot(tb, R / np.max(R), label=rf"$\sigma= {variance}$", color=colors[i])


def Lorentz_PSD(theta):
    """PSD of a single Lorentz pulse with duration time td = 1"""
    return 2 * np.pi * np.exp(-2 * np.abs(theta))


def spectra_analytical(omega, gamma, A_rms, A_mean, sigma):
    nu = sigma * gamma
    Omega = omega / gamma
    I_2 = 1 / (2 * np.pi)
    first_term = gamma * A_rms**2 * I_2 * Lorentz_PSD(omega)
    second_term = (
        gamma**1
        * A_mean**2
        * I_2
        * Lorentz_PSD(omega)
        * np.sinh(nu**2 * Omega**2 / 2)
        / (np.cosh(nu**2 * Omega**2 / 2) - np.cos(Omega))
    )
    return 2 * (first_term + second_term)


# PSD = PSD_periodic_arrivals(2 * np.pi * f, td=1, gamma=0.2, A_rms=1, A_mean=1, dt=0.01)
PSD = spectra_analytical(2 * np.pi * f, gamma=0.2, A_rms=1, A_mean=1, sigma=0.05 / 0.2)
ax1.semilogy(f, PSD, "--k")  # , label=r"$S_{{\Phi}}(\tau_\mathrm{d} f)$")
PSD = spectra_analytical(2 * np.pi * f, gamma=0.2, A_rms=1, A_mean=1, sigma=0.1 / 0.2)
ax1.semilogy(f, PSD, "--k")  # , label=r"$S_{{\Phi}}(\tau_\mathrm{d} f)$")
PSD = spectra_analytical(2 * np.pi * f, gamma=0.2, A_rms=1, A_mean=1, sigma=1 / 0.2)
ax1.semilogy(f, PSD, "--k")  # , label=r"$S_{{\Phi}}(\tau_\mathrm{d} f)$")
# t = np.linspace(0, 50, 1000)
# R_an = autocorr_periodic_arrivals(t, 0.2, 1, 1)
# ax2.plot(t, R_an, "--k", label=r"$R_{\widetilde{\Phi}}(t/\tau_\mathrm{d})$")

ax1.set_xlabel(r"$\tau_\mathrm{d} f$")
ax1.set_ylabel(r"$S_{{\Phi}}(\tau_\mathrm{d} f)$")
ax1.set_xlim(-0.03, 1)
ax1.set_ylim(1e-4, 1e2)
ax1.legend()

ax2.set_xlim(0, 50)
ax2.set_xlabel(r"$t/\tau_\mathrm{d}$")
ax2.set_ylabel(r"$R_{\widetilde{\Phi}}(t/\tau_\mathrm{d})$")
ax2.legend()
cosmoplots.change_log_axis_base(ax1, "y", base=10)

fig_PSD.savefig("PSD_guassian_jitter.png", bbox_inches="tight")
fig_AC.savefig("AC_gaussian_jitter.png", bbox_inches="tight")

plt.show()

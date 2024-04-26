import numpy as np
from scipy import signal
from support_functions import *
import superposedpulses.forcing as frc
import superposedpulses.point_model as pm
import superposedpulses.pulse_shape as ps
from closedexpressions import PSD_periodic_arrivals, autocorr_periodic_arrivals
import cosmoplots
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use("cosmoplots.default")

fig_PSD = plt.figure()
ax1 = fig_PSD.gca()
cosmoplots.change_log_axis_base(ax1, "y")
fig_AC = plt.figure()
ax2 = fig_AC.gca()
colors = ["tab:blue", "tab:orange", "tab:olive"]

Beta =[1000,100,10]
beta_label = [r"$10^3$", r"$10^2$", r"$10$"]
gamma = 0.2
dt = 1e-2

class ForcingGammaDistribution(frc.ForcingGenerator):
    def __init__(self, beta):
        self.beta = beta

    def get_forcing(self, times: np.ndarray, gamma: float) -> frc.Forcing:
        total_pulses = int(max(times) * gamma)
        waiting_times = (
            np.random.gamma(
                self.beta, scale=(gamma * self.beta) ** (-1), size=total_pulses
            )
            * 100  # multiplied with inverse dt
        )
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


model = pm.PointModel(gamma=gamma, total_duration=100000, dt=dt)
model.set_pulse_shape(ps.LorentzShortPulseGenerator(tolerance=1e-5))

for i, beta in enumerate(Beta):
    model.set_custom_forcing_generator(ForcingGammaDistribution(beta=beta))

    T, S = model.make_realization()
    forcing = model.get_last_used_forcing()
    amp = forcing.amplitudes

    S_norm = (S - S.mean())

    f, Pxx = signal.welch(x=S_norm, fs=1./dt, nperseg=S.size / 30)
    ax1.plot(f, Pxx, label=rf"$\beta =$" + beta_label[i], color=colors[i])

    tb, R = corr_fun(S_norm, S_norm, dt=dt, norm=False, biased=True, method="auto")
    # divide by max to show normalized Phi
    ax2.plot(tb[abs(tb)<50], R[abs(tb)<50]/np.max(R), label=rf"$\beta =$" + beta_label[i], color=colors[i])

def Lorentz_PSD(theta):
    """PSD of a single Lorentz pulse with duration time td = 1"""
    return 2 * np.pi * np.exp(-2 * np.abs(theta))


def spectra_analytical(omega, gamma, A_rms, A_mean, beta):
    I_2 = 1 / (2 * np.pi)
    first_term = gamma * A_rms**2 * I_2 * Lorentz_PSD(omega)
    second_term = (
        gamma
        * A_mean**2
        * I_2
        * Lorentz_PSD(omega)
        * np.real(((1-1.j*omega/(gamma*beta))**beta+1)/((1-1.j*omega/(gamma*beta))**beta-1))
    )
    return 2 * (first_term + second_term)


for label, ls, beta in zip([r"$S_{{\Phi}}(\tau_\mathrm{d} f)$",None, None],
                            ['--','-.',':'],
                            Beta):
    PSD = spectra_analytical(
        2 * np.pi * f, gamma=gamma, A_rms=1, A_mean=1, beta=beta 
    )
    ax1.plot(f, PSD, ls+"k",label=label)
#PSD = PSD_periodic_arrivals(2 * np.pi * f, td=1, gamma=0.2, A_rms=1, A_mean=1, dt=0.01)
#ax1.plot(f, PSD, "--k", label=r"$S_{\widetilde{\Phi}}(\tau_\mathrm{d} f)$")

#t = np.linspace(0, 50, 1000)
#R_an = autocorr_periodic_arrivals(t, 0.2, 1, 1)
#ax2.plot(t, R_an, "--k", label=r"$R_{\widetilde{\Phi}}(t/\tau_\mathrm{d})$")

def Lorentz_AC_basic(t):
    return 4/(4+t**2)

tb = np.linspace(0,50,1000)
ax2.plot(tb, Lorentz_AC_basic(tb),':k',label=r'$\rho_\phi(t/\tau_\mathrm{d})$')

ax1.set_xlim(-0.03, 1)
ax1.set_ylim(1e-5, 1e1)
ax1.set_xlabel(r"$\tau_\mathrm{d} f$")
ax1.set_ylabel(r"$S_{\widetilde{\Phi}}(\tau_\mathrm{d} f)$")
ax1.legend()

ax2.set_xlim(0, 50)
ax2.set_xlabel(r"$t/\tau_\mathrm{d}$")
ax2.set_ylabel(r"$R_{\widetilde{\Phi}}(t/\tau_\mathrm{d})$")
ax2.legend()

fig_PSD.savefig("PSD_different_gamma.eps", bbox_inches="tight")
fig_AC.savefig("AC_different_gamma.eps", bbox_inches="tight")

"""
Figures for the Rayleigh-Bénard convection, both raw time series and power spectra, plus stochastic model fits.
"""

import sys

sys.path.append("..")

import numpy as np
from plasmapy.analysis.time_series.conditional_averaging import ConditionalEvents
from scipy import signal
import support_functions as sf
import fppanalysis as fa
import matplotlib.pyplot as plt
import cosmoplots

plt.style.use("cosmoplots.default")

# Match \mathcal{E} in text
# OBS! In these notes, we call the kinetic energy K, but the figure labels use E since K is taken.
plt.rcParams["font.family"] = "serif"

mu_list = ["1.6e-3", "1e-4"]


class MuOpts:
    def __init__(self, mu):
        if mu == mu_list[0]:
            self.mu = mu_list[0]
            self.savename = "1_6e-3"
            self.label = r"$1.6\times 10^{-3}$"
            self.color = "C1"
            self.ls = "--"
            self.wait_min = 60
            self.tstart = 20000
            self.ts_lim = (0, 2000, None, None)
            self.spectra_lim = (0, 0.1, 1e-6, None)
        elif mu == mu_list[1]:
            self.mu = mu_list[1]
            self.savename = "1e-4"
            self.label = r"$10^{-4}$"
            self.color = "C2"
            self.ls = ":"
            self.wait_min = 300
            self.tstart = 70000
            self.ts_lim = (0, 2000, -1, 14)
            self.spectra_lim = (0, 3e-2, 1e-6, None)


def plot_RB(fit=False):
    if fit:
        fig, ax = cosmoplots.figure_multiple_rows_columns(3, 2)
        figav, axav = cosmoplots.figure_multiple_rows_columns(2, 2)
    else:
        fig, ax = cosmoplots.figure_multiple_rows_columns(2, 2)

    for i, mu in enumerate(mu_list):
        Mu = MuOpts(mu)
        K = np.load("./RB_data/K_" + Mu.mu + "_data.npy")
        time = np.load("./RB_data/time_" + Mu.mu + "_data.npy")
        dt = time[1] - time[0]

        nK = (K - np.mean(K)) / np.std(K)
        fK, PK = signal.welch(nK, 1 / dt, nperseg=len(nK) / 4)

        ax[i].plot(time - Mu.tstart, nK)
        ax[i].set_xlabel(r"$t$")
        ax[i].set_ylabel(r"$\widetilde{\mathcal{E}}$")
        ax[i].axis(Mu.ts_lim)
        if Mu.mu == mu_list[1]:
            ax[i].set_yticks(range(0, 15, 3))

        cosmoplots.change_log_axis_base(ax[i + 2], "y")
        ax[i + 2].plot(fK[1:], PK[1:] * K.std() ** 2)
        ax[i + 2].set_ylabel(
            r"$\mathcal{E}_\mathrm{rms}^2 S_{\widetilde{\mathcal{E}}}\left( f \right)$"
        )
        ax[i + 2].set_xlabel(r"$f$")
        ax[i + 2].axis(Mu.spectra_lim)

        if fit:
            CoEv = ConditionalEvents(
                signal=K,
                time=time,
                lower_threshold=K.mean() + K.std(),
                distance=Mu.wait_min,
                remove_non_max_peaks=True,
            )

            fitfile = open("fitdata_" + Mu.savename + ".txt", "w")
            fitfile.write("<K>={}, K_rms={}\n".format(np.mean(K), np.std(K)))

            K_fit, pulse = sf.create_fit(
                dt, time, CoEv
            )  # pulse contains (time_kern, kern, (td, lam))
            nK_fit = (K_fit - np.mean(K_fit)) / np.std(K_fit)

            fitfile.write(
                "<K_fit>={}, K_fit_rms={}\n".format(np.mean(K_fit), np.std(K_fit))
            )
            fitfile.write(
                "td={}, lam={}\n <tw>={},tw_rms={}\n <A>={}, A_rms={}".format(
                    pulse[2][0],
                    pulse[2][1],
                    CoEv.waiting_times.mean(),
                    CoEv.waiting_times.std(),
                    CoEv.peaks.mean(),
                    CoEv.peaks.std(),
                )
            )
            fitfile.close()

            axav[i].plot(pulse[0], pulse[1], "k")
            axav[i].plot(
                CoEv.time,
                CoEv.average / max(CoEv.average),
                c=Mu.color,
                ls=Mu.ls,
                label=r"$\mu=$" + Mu.label,
            )
            axav[i].set_xlim([-50, 50])
            axav[i].set_xlabel(r"$t$")
            axav[i].set_ylabel(
                r"$\langle \mathcal{E}(t-s) | \mathcal{E}(s)=\mathcal{E}_\mathrm{max}\rangle$"
            )
            axav[i].legend()

            pdf, _, x = fa.distribution(
                CoEv.peaks / np.mean(CoEv.peaks), 32, kernel=True
            )
            axav[2].plot(x, pdf, c=Mu.color, ls=Mu.ls)
            axav[2].set_xlabel(r"$A/\langle A\rangle$")
            axav[2].set_ylabel(r"$P(A/\langle A\rangle)$")

            pdf, _, x = fa.distribution(
                CoEv.waiting_times / np.mean(CoEv.waiting_times), 32, kernel=True
            )
            axav[3].plot(x, pdf, c=Mu.color, ls=Mu.ls)
            axav[3].set_xlabel(r"$\tau_\mathrm{w}/\langle\tau_\mathrm{w}\rangle$")
            axav[3].set_ylabel(r"$P(\tau_\mathrm{w}/\langle\tau_\mathrm{w}\rangle)$")

            ax[i].plot(
                time + (CoEv.arrival_times[0] - time[0]) - Mu.tstart,
                nK_fit,
                "--",
                c=Mu.color,
            )

            f_fit, PK_fit = signal.welch(nK_fit, 1 / dt, nperseg=int(len(nK_fit) / 4))
            ax[i + 2].plot(f_fit, PK_fit * K_fit.std() ** 2, "--", c=Mu.color)
            ax[i + 2].plot(
                f_fit,
                sf.spectrum_gauss(
                    f_fit,
                    pulse[2][0],
                    pulse[2][1],
                    CoEv.peaks.mean(),
                    CoEv.peaks.std(),
                    CoEv.waiting_times.mean(),
                    CoEv.waiting_times.std(),
                ),
                "k--",
            )

            cosmoplots.change_log_axis_base(ax[i + 4], "y")
            ax[i + 4].plot(
                f_fit,
                sf.est_wait_spectrum_ECF(f_fit, CoEv.waiting_times),
                c=Mu.color,
                label=r"$\mathrm{ECF}$",
            )
            ax[i + 4].plot(
                f_fit,
                sf.spectrum_gauss_renewal_part(
                    f_fit, CoEv.waiting_times.mean(), CoEv.waiting_times.std()
                ),
                "k--",
                label=r"$\mathrm{Normal}$",
            )
            ax[i + 4].set_ylabel(
                r"$\mathcal{E}_\mathrm{rms}^2 S_{\widetilde{\mathcal{E}}}\left( f \right)$"
            )
            ax[i + 4].set_xlabel(r"$f$")
            ax[i + 4].legend()
            ax[i + 4].set_xlim(Mu.spectra_lim[:2])
    if fit:
        figav.savefig("Kav_fit.eps")
        fig.savefig("K_fit.eps")
    else:
        fig.savefig("K_nofit.eps")


plot_RB(False)
plot_RB(True)

"""
Comparison of analytic and numerical spectra for the process with gaussian jitter.
"""

import numpy as np
import matplotlib.pyplot as plt
import cosmoplots

plt.style.use("cosmoplots.default")


def cf_norm(omega, tw, wrms):
    return np.exp(-(wrms**2) * omega**2)


def spectrum_waiting_time_part(omega, charfun):
    cf = charfun(omega)

    jitter = 1 - cf
    periodic = np.zeros(omega.size)
    for o in range(omega.size):
        if (omega[o] % (2 * np.pi)) < 0.02:
            periodic[o] = cf[o] * 10000

    return jitter + periodic


def psd_norm_num(wrms):
    dname = "psd_norm_jitter_fmax_5.0_w_{:.1e}.npz".format(wrms)
    D = np.load(dname)
    return D["freq"], D["psd"]


linestyle = ["-", "--"]
color = ["C0", "k"]
labels = [
    r"$\mathrm{Normal\,num.}$",
    r"$\mathrm{Normal\,an.}$",
]

wrms_label = [r"$1/10$", r"$1$", r"$5$"]
Wrms = [1e-1, 1, 5]

tw = 1.0

dt = 0.01
f = np.arange(1, int(10 / dt) + 1) * dt
omega = 2 * np.pi * f

rows = 3
columns = 1
fig, ax = cosmoplots.figure_multiple_rows_columns(rows, columns)

ax[2].set_xscale("log")
for row in range(rows):
    F, Pn = psd_norm_num(Wrms[row])
    pa = spectrum_waiting_time_part(omega, lambda o: cf_norm(o, 1, Wrms[row]))
    if row == 1:
        ax[row].semilogy(F, Pn, c=color[0], label=labels[0])
        ax[row].semilogy(f, pa, c=color[1], ls=linestyle[1], label=labels[1])

    else:
        ax[row].semilogy(F, Pn, c=color[0])
        ax[row].semilogy(f, pa, c=color[1], ls=linestyle[1])
    ax[row].plot(F[Pn > 1.5], Pn[Pn > 1.5], c=color[0], ls=" ", marker="o")
    ax[row].plot(f[pa > 1.5], pa[pa > 1.5], c=color[0], ls=" ", marker="o")

    ax[row].set_ylabel(r"$S_\Phi$")
    ax[row].set_xlabel(r"$\langle w \rangle f$")
    if row == 2:
        ax[2].set_xlim(1e-2, 5)
    else:
        ax[row].set_xlim(0, 5)
    if row == 0:
        ax[row].set_ylim(1e-2, 1e4)
    else:
        ax[row].set_ylim(1e-2, 1e2)
    ax[row].text(
        0.8,
        0.1,
        r"$w_\mathrm{rms}/\langle w \rangle=\,$" + wrms_label[row],
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax[row].transAxes,
    )
    cosmoplots.change_log_axis_base(ax[row], base=10)
ax[1].legend()


fig.savefig("PSD_compare_jitter.eps")

"""
Comparison of spectra for the process with fGn waiting time distributions. <w>/w_rms=1
"""

import numpy as np
import matplotlib.pyplot as plt
import cosmoplots

plt.style.use("cosmoplots.default")


def cf_norm(omega, tw, wrms):
    return np.exp(1.0j * tw * omega - 0.5 * wrms**2 * omega**2)


def spectrum_waiting_time_part(omega, charfun):
    cf = charfun(omega)
    return np.real((1 + cf) / (1 - cf))


def psd_fGn_num(H):
    dname = "psd_w_fGn_fmax_2.0_H_{:.1e}.npz".format(float(H))
    D = np.load(dname)
    return D["freq"], D["psd"]


linestyle = ["-", "--", "-.", ":"]
labels = [
    r"$\mathrm{Normal\,num.}$",
    r"$\mathrm{Inverse\, Gamma}$",
    r"$\mathrm{Gamma}$",
]

H_label = [r"$H=1/10$", r"$H=1/2$", r"$H=9/10$"]
H = [0.1, 0.5, 0.9]

tw = 1.0

dt = 0.001
f = np.arange(1, int(10 / dt) + 1) * dt
omega = 2 * np.pi * f

rows = 1
columns = 1
fig, ax = cosmoplots.figure_multiple_rows_columns(
    rows,
    columns,
    labels=[
        "",
    ],
)
ax[0].set_xscale("log")
for h, lab in zip(H, H_label):
    ax[0].semilogy(*psd_fGn_num(h), label=lab)

ax[0].semilogy(
    f,
    spectrum_waiting_time_part(omega, lambda o: cf_norm(o, 1, 1)),
    "k:",
    label=r"$\mathrm{Normal\,an.}$",
)
ax[0].semilogy(
    f[f < 0.1],
    f[f < 0.1] ** (-4 / 5) / 2,
    c="C2",
    ls="--",  # , label=r"$f^{-4/5}$"
)
ax[0].semilogy(
    f[f < 0.1], f[f < 0.1] ** (4 / 5), c="C0", ls="--"
)  # , label=r"$f^{4/5}$")
ax[0].set_ylabel(r"$\mathrm{Re}\left[ (1 + \psi_w)/(1-\psi_w) \right]$")
ax[0].set_xlabel(r"$\langle w \rangle f$")
ax[0].set_xlim(1e-3, 2)
ax[0].set_ylim(1e-3, 1e3)
cosmoplots.change_log_axis_base(ax[0], base=10)
ax[0].legend()


fig.savefig("psdcomparefgn.eps")

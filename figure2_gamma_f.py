import matplotlib.pyplot as plt
from uit_scripts.shotnoise import make_signal
import numpy as np
from scipy import signal
from uit_scripts.plotting import figure_defs
from support_functions import *

axes_size = figure_defs.set_rcparams_aip(plt.rcParams, num_cols=1, ls="thin")
fig_PSD = plt.figure()
ax1 = fig_PSD.add_axes(axes_size)
fig_AC = plt.figure()
ax2 = fig_AC.add_axes(axes_size)


T, S, amp, ta = make_signal(
    gamma=0.2,
    K=10000,
    dt=0.01,
    convolve=True,
    mA=1,
    ampta=True,
    kerntype="lorentz",
    TWdist="gam",
    TWkappa=1000,
)


S_norm = (S - S.mean()) / S.std()

f, Pxx = signal.welch(x=S_norm, fs=100, nperseg=S.size / 30)

ax1.semilogy(f, Pxx, label=r"$\beta = 1000$")
PSD = PSD_periodic_arrivals(
    2 * np.pi * f, td=1, gamma=0.2, Arms=amp.std(), Am=np.mean(amp), S=S
)

ax1.set_xlim(-0.2, 12)
ax1.set_ylim(1e-14, 1e3)

ax1.set_xlabel(r"$f$")
ax1.set_ylabel(r"$S_{\widetilde{\Phi}}(f)$")

tb, R = corr_fun(S_norm, S_norm, dt=0.01, norm=False, biased=True, method="auto")

ax2.plot(tb, R, label=r"$\beta = 1000$")

t, R_an = calculate_R_an(1, 1, 0.2)


T, S, amp, ta = make_signal(
    gamma=0.2,
    K=10000,
    dt=0.01,
    convolve=True,
    mA=1,
    ampta=True,
    kerntype="lorentz",
    TWdist="gam",
    TWkappa=100,
)


S_norm = (S - S.mean()) / S.std()


f, Pxx = signal.welch(x=S_norm, fs=100, nperseg=S.size / 30)

ax1.semilogy(f, Pxx, label=r"$\beta = 100$")
PSD = PSD_periodic_arrivals(
    2 * np.pi * f, td=1, gamma=0.2, Arms=amp.std(), Am=np.mean(amp), S=S
)

ax1.set_xlim(-0.2, 12)
ax1.set_ylim(1e-14, 1e3)

ax1.set_xlabel(r"$f$")
ax1.set_ylabel(r"$S_{\widetilde{\Phi}}(f)$")

tb, R = corr_fun(S_norm, S_norm, dt=0.01, norm=False, biased=True, method="auto")

ax2.plot(tb, R, label=r"$\beta = 100$")

t, R_an = calculate_R_an(1, 1, 0.2)


T, S, amp, ta = make_signal(
    gamma=0.2,
    K=10000,
    dt=0.01,
    convolve=True,
    mA=1,
    ampta=True,
    kerntype="lorentz",
    TWdist="gam",
    TWkappa=10,
)


S_norm = (S - S.mean()) / S.std()

f, Pxx = signal.welch(x=S_norm, fs=100, nperseg=S.size / 30)

ax1.semilogy(f, Pxx, label=r"$\beta = 10$")
PSD = PSD_periodic_arrivals(
    2 * np.pi * f, td=1, gamma=0.2, Arms=amp.std(), Am=np.mean(amp), S=S
)


ax1.semilogy(f, PSD, "--k", label=r"$S_{\widetilde{\Phi}}(f)$")

tb, R = corr_fun(S_norm, S_norm, dt=0.01, norm=False, biased=True, method="auto")

ax2.plot(tb, R, label=r"$\beta = 10$")

t, R_an = calculate_R_an(1, 1, 0.2)

ax2.plot(t, R_an, "--k", label=r"$R_{\widetilde{\Phi}}(t)$")

ax1.legend()
ax1.set_xlim(-0.03, 1)
ax1.set_ylim(1e-4, 1e2)

ax2.set_xlim(0, 50)
ax2.set_xlabel(r"$t$")
ax2.set_ylabel(r"$R_{\widetilde{\Phi}}(t)$")
ax2.legend()
# fig_PSD.savefig('PSD_different_gamma.eps', bbox_inches = 'tight')
# fig_AC.savefig('AC_different_gamma.eps', bbox_inches = 'tight')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import cosmoplots
from fppanalysis import cond_av

from plot_lorentz_time_series import (
    skewed_lorentz,
)

axes_size = cosmoplots.set_rcparams_dynamo(plt.rcParams, num_cols=1, ls="thin")

K = np.load("K.npy")
U = np.load("U.npy")
time = np.load("K_time.npy")

# remove first event in time series
K = K[600:]
U = U[600:]
time = time[600:]

# K = (K - np.mean(K)) / np.std(K)
_, s_av, _, t_av, peaks, wait = cond_av(K, time, smin=2.5, window=True, delta=0.08)

# plt.scatter(peaks[:-1], peaks[1:])
# plt.xlabel(r"$A_n$")
# plt.ylabel(r"$A_{n+1}$")
# plt.savefig("A_A+1_K.pdf", bbox_inches="tight")
# plt.show()
#
# plt.scatter(peaks[:-1], wait[1:])
# plt.xlabel(r"$A_n$")
# plt.ylabel(r"$\tau_w$")
# plt.savefig("A_tau_K.pdf", bbox_inches="tight")
# plt.show()

kern = skewed_lorentz(t_av, t_av[1] - t_av[0], 0.0, 0.002, m=0)
plt.plot(t_av, s_av/np.max(s_av))
plt.plot(t_av,kern/np.max(kern))
plt.xlabel(r"$t$")
plt.ylabel(r"$\left<\widetilde{K}\right>/\textrm{max}\left<\widetilde{K}\right>$")
# plt.savefig("CA_K.pdf", bbox_inches="tight")
plt.show()
#
# plt.hist(wait, 32)
# plt.xlabel(r'$\tau_w$')
# plt.ylabel(r'$P(\tau_w)$')
# plt.savefig("tau_K.pdf", bbox_inches="tight")
# plt.show()
#
# plt.hist(peaks, 32)
# plt.xlabel(r'$peaks$')
# plt.ylabel(r'$P(peaks)$')
# plt.savefig("peaks_K.pdf", bbox_inches="tight")
# plt.show()
#
# plt.plot(time, K)
# plt.xlabel(r'$t$')
# plt.ylabel(r'$\widetilde{K}$')
# plt.savefig("K_norm.pdf", bbox_inches="tight")
# plt.show()

# U = (U - np.mean(U)) / np.std(U)
_, s_av, _, t_av, peaks, wait = cond_av(U, time, smin=0, window=True, delta=0.08)

# plt.scatter(peaks[:-1], peaks[1:])
# plt.xlabel(r"$A_n$")
# plt.ylabel(r"$A_{n+1}$")
# plt.savefig("A_A+1_U.pdf", bbox_inches="tight")
# plt.show()
#
# plt.scatter(peaks[:-1], wait[1:])
# plt.xlabel(r"$A_n$")
# plt.ylabel(r"$\tau_w$")
# plt.savefig("A_tau_U.pdf", bbox_inches="tight")
# plt.show()

kern = skewed_lorentz(t_av, t_av[1] - t_av[0], -0.999, 0.01, m= 0.0)
plt.plot(t_av, s_av/np.max(s_av))
plt.plot(t_av+0.005,kern/np.max(kern))
plt.xlabel(r"$t$")
plt.ylabel(r"$\left<\widetilde{K}\right>/\textrm{max}\left<\widetilde{K}\right>$")
plt.savefig("CA_U.pdf", bbox_inches="tight")
plt.show()
#
# plt.hist(wait, 32)
# plt.xlabel(r'$\tau_w$')
# plt.ylabel(r'$P(\tau_w)$')
# plt.savefig("tau_U.pdf", bbox_inches="tight")
# plt.show()
#
# plt.hist(peaks, 32)
# plt.xlabel(r'$peaks$')
# plt.ylabel(r'$P(peaks)$')
# plt.savefig("peaks_U.pdf", bbox_inches="tight")
# plt.show()
#
# plt.plot(time, U)
# plt.xlabel(r'$t$')
# plt.ylabel(r'$\widetilde{U}$')
# plt.savefig("U_norm.pdf", bbox_inches="tight")
# plt.show()

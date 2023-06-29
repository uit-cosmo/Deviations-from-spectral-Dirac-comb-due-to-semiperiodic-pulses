import numpy as np
from fppanalysis import cond_av
from scipy import signal
import matplotlib.pyplot as plt
from fit_function_RB_model import create_fit_K
import cosmoplots


axes_size = cosmoplots.set_rcparams_dynamo(plt.rcParams, num_cols=1, ls="thin")


def double_exp(tkern, lam, td):
    kern = np.zeros(tkern.size)
    kern[tkern < 0] = np.exp(tkern[tkern < 0] / lam / td)
    kern[tkern >= 0] = np.exp(-tkern[tkern >= 0] / (1 - lam) / td)
    return kern


K = np.load("./RB_data/K_1e-4_data.npy")
time = np.load("./RB_data/time_1e-4_data.npy")

dt = time[1] - time[0]

_, K_av, _, t_av, peaks, wait = cond_av(K, time, smin=1, window=True, delta=200)

wait = wait[wait > 200]
plt.hist(wait / np.mean(wait), 32, density=True)
plt.xlabel(r"$\tau_w/\langle\tau_w\rangle$")
plt.ylabel(r"$P(\tau_w/\langle\tau_w\rangle)$")
plt.savefig("P(tau)_1e-4.eps", bbox_inches="tight")
plt.show()

K = (K - np.mean(K)) / np.std(K)
fK, PK = signal.welch(K, 1 / dt, nperseg=len(K) / 4)

K_fit, symbols, _, _ = create_fit_K(
    fK, dt, K, time, "exp", td=10, shuffled=False, lam=0.5
)

plt.plot(time, K)
plt.plot(time, K_fit, "--")
plt.xlabel(r"$t$")
plt.ylabel(r"$\widetilde{K}$")
plt.xlim(70000, 72000)
plt.savefig("K_1e-4.eps", bbox_inches="tight")
plt.show()


f, PK_fit = signal.welch(K_fit, 1 / dt, nperseg=len(K_fit) / 4)

plt.semilogy(fK, PK)
plt.semilogy(f, PK_fit, "--")
plt.xlabel(r"$f$")
plt.ylabel(r"$S_{\widetilde{K}}\left( f \right)$")
plt.xlim(-0.003, 0.03)
plt.ylim(1e-1, None)
plt.savefig("S(K)_1e-4.eps", bbox_inches="tight")
plt.show()

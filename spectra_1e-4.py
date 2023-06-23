import numpy as np
from fppanalysis import cond_av
from scipy import signal
import matplotlib.pyplot as plt
from fit_function_RB_model import create_fit_K


def double_exp(tkern, lam, td):
    kern = np.zeros(tkern.size)
    kern[tkern < 0] = np.exp(tkern[tkern < 0] / lam / td)
    kern[tkern >= 0] = np.exp(-tkern[tkern >= 0] / (1 - lam) / td)
    return kern


K = np.load("K_1e-4.npy")
time = np.load("K_time_1e-4.npy")
U = np.load("U_1e-4.npy")
dt = time[1] - time[0]

K = K[2000:]
U = U[2000:]
time = time[2000:]

plt.plot(time, K)
plt.plot(time, U)
plt.show()

_, K_av, _, t_av, peaks, wait = cond_av(K, time, smin=1, window=True, delta=200)
kern = double_exp(t_av, 0.5, 10)

plt.plot(t_av, K_av / np.max(K_av))
plt.plot(t_av, kern / np.max(kern))
plt.show()

_, U_av, _, t_av, peaks, wait = cond_av(U, time, smin=0, window=True, delta=200)
kern = double_exp(t_av, 0.1, 500)

plt.plot(t_av, U_av / np.max(U_av))
plt.plot(t_av, kern / np.max(kern))
plt.show()

plt.hist(peaks, 32)
plt.xlabel('peaks')
plt.ylabel('P(peaks)')
plt.show()

plt.hist(wait, 32)
plt.xlabel(r'$\tau_w$')
plt.ylabel(r'$P(\tau_w)$')
plt.show()

K = (K - np.mean(K)) / np.std(K)
U = (U - np.mean(U)) / np.std(U)

fK, PK = signal.welch(K, 1 / dt, nperseg=len(K) / 4)
fU, PU = signal.welch(U, 1 / dt, nperseg=len(U) / 4)

plt.semilogy(fK, PK)
plt.semilogy(fU, PU)
plt.show()


K_fit, symbols, _, _ = create_fit_K(
    fK, dt, K, time, "exp", td=10, shuffled=False, lam=0.5
)
print(K_fit)

plt.plot(time, K)
plt.plot(time, K_fit, "--")
plt.xlabel(r"$t$")
plt.ylabel(r"$\widetilde{K}$")
plt.show()


f, PK_fit = signal.welch(K_fit, 1 / dt, nperseg=len(K_fit) / 4)

plt.semilogy(fK, PK)
plt.semilogy(f, PK_fit, "--")
plt.xlabel(r"$f$")
plt.ylabel(r"$S_{K}\left( f \right)$")
# plt.xlim(-10, 200)
# plt.ylim(1e-5, None)
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from fit_function_RB_model import create_fit_K
import cosmoplots


axes_size = cosmoplots.set_rcparams_dynamo(plt.rcParams, num_cols=1, ls="thin")

K = np.load("K.npy")
U = np.load("U.npy")
time = np.load("K_time.npy")

# remove first event in time series
K = K[600:]
U = U[600:]
time = time[600:]
dt = time[1] - time[0]

K = (K - np.mean(K)) / np.std(K)

f, Pxx = signal.welch(K, 1 / dt, nperseg=len(K) / 5)

time_series_fit, symbols, _, forcing = create_fit_K(f, dt, K, time, "exp", 0.005, shuffled = False)

f, Pxx_fit = signal.welch(time_series_fit, 1 / dt, nperseg=len(time_series_fit) / 5)


plt.plot(time, K)
plt.plot(time, time_series_fit, "--")
plt.xlabel(r"$t$")
plt.ylabel(r"$\widetilde{K}$")
plt.xlim(3,4)
plt.savefig("K_fit.png", bbox_inches="tight")
plt.show()

plt.semilogy(f, Pxx)
plt.loglog(f, Pxx_fit, "--")
plt.xlabel(r"$f$")
plt.ylabel(r"$S_{K}\left( f \right)$")
plt.xlim(1, 200)
plt.ylim(1e-5, None)
plt.savefig("S_K_loglog.png", bbox_inches="tight")
plt.show()

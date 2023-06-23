import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

K = np.load("K_1e-4.npy")
time = np.load("K_time_1e-4.npy")
U = np.load("U_1e-4.npy")
dt = time[1] - time[0]

K = K[1000:]
U = U[1000:]
time = time[1000:]

plt.plot(time, K)
plt.plot(time, U)
plt.show()

fK, PK = signal.welch(K, 1 / dt, nperseg=len(K) / 5)
fU, PU = signal.welch(U, 1 / dt, nperseg=len(U) / 5)

plt.semilogy(fK, PK)
plt.semilogy(fU, PU)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import cosmoplots

axes_size = cosmoplots.set_rcparams_dynamo(plt.rcParams, num_cols=1, ls="thin")

K = np.load('K.npy')
U = np.load('U.npy')
time = np.load('K_time.npy')
dt = time[1] - time[0]

# plt.plot(time, K)
# plt.xlabel(r'$t$')
# plt.ylabel(r'$K$')
# plt.show() 31 f, Pxx = signal.welch(K, 1/dt, nperseg=len(K) / 1)

f, Pxx = signal.welch(K, 1/dt, nperseg=len(K) / 5)
plt.semilogy(f, Pxx)
plt.xlabel(r"$f$")
plt.ylabel(r"$S_{K}\left( f \right)$")
plt.xlim(-10,200)
plt.ylim(1e3,None)
plt.show()


f, Pxx = signal.welch(U, 1/dt, nperseg=len(K) / 5)
plt.semilogy(f, Pxx)
plt.xlabel(r"$f$")
plt.ylabel(r"$S_{U}\left( f \right)$")
plt.xlim(-10,200)
plt.ylim(1e3,None)
plt.show()

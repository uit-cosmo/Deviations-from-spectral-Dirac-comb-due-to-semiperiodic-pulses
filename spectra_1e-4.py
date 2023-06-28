import numpy as np
from fppanalysis import cond_av
from scipy import signal
import matplotlib.pyplot as plt
from fit_function_RB_model import create_fit_K, create_fit_U
import cosmoplots


axes_size = cosmoplots.set_rcparams_dynamo(plt.rcParams, num_cols=1, ls="thin")


def double_exp(tkern, lam, td):
    kern = np.zeros(tkern.size)
    kern[tkern < 0] = np.exp(tkern[tkern < 0] / lam / td)
    kern[tkern >= 0] = np.exp(-tkern[tkern >= 0] / (1 - lam) / td)
    return kern


K = np.load("K_1e-4.npy")
time = np.load("K_time_1e-4.npy")
# U = np.load("U_1e-4.npy")
dt = time[1] - time[0]

K_2 = np.load("K_1e-4_2.npy")
time_2 = np.load("K_time_1e-4_2.npy")

K_3 = np.load("K_1e-4_3.npy")
time_3 = np.load("K_time_1e-4_3.npy")

K_4 = np.load("K_1e-4_4.npy")
time_4 = np.load("K_time_1e-4_4.npy")

K = np.append(K, K_2)
time = np.append(time, time_2)

K = np.append(K, K_3)
time = np.append(time, time_3)

K = np.append(K, K_4)
time = np.append(time, time_4)

K = K[2000:]
# U = U[2000:]
time = time[2000:]

plt.plot(time, K)
# plt.plot(time, U)
plt.show()

_, K_av, _, t_av, peaks, wait = cond_av(K, time, smin=1, window=True, delta=200)
kern = double_exp(t_av, 0.5, 10)

print(f"Mean wait: {np.mean(wait)}")
print(f"std wait: {np.std(wait)}")

plt.plot(t_av, K_av / np.max(K_av))
plt.plot(t_av, kern / np.max(kern))
plt.show()

# _, U_av, _, t_av, peaks, wait = cond_av(U, time, smin=0, window=True, delta=200)
# kern = double_exp(t_av, 0.1, 500)

# plt.plot(t_av, U_av / np.max(U_av))
# plt.plot(t_av, kern / np.max(kern))
# plt.show()

# plt.hist(peaks, 32)
# plt.xlabel("peaks")
# plt.ylabel("P(peaks)")
# plt.show()

wait = wait[wait > 200]
plt.hist(wait / np.mean(wait), 32, density=True)
plt.xlabel(r"$\tau_w/\langle\tau_w\rangle$")
plt.ylabel(r"$P(\tau_w/\langle\tau_w\rangle)$")
plt.savefig("P(tau)_1e-4.eps", bbox_inches="tight")
plt.show()

K = (K - np.mean(K)) / np.std(K)
# U = (U - np.mean(U)) / np.std(U)

fK, PK = signal.welch(K, 1 / dt, nperseg=len(K) / 4)
# fU, PU = signal.welch(U, 1 / dt, nperseg=len(U) / 4)

plt.semilogy(fK, PK)
# plt.semilogy(fU, PU)
plt.show()


K_fit, symbols, _, _ = create_fit_K(
    fK, dt, K, time, "exp", td=10, shuffled=False, lam=0.5
)

plt.plot(time, K)
# plt.plot(time, K_tit, "--")
plt.xlabel(r"$t$")
plt.ylabel(r"$\widetilde{K}$")
plt.xlim(70000, 72000)
plt.savefig("K_1e-4.eps", bbox_inches="tight")
plt.show()


f, PK_fit = signal.welch(K_fit, 1 / dt, nperseg=len(K_fit) / 4)

plt.semilogy(fK, PK)
# plt.semilogy(f, PK_fit, "--")
plt.xlabel(r"$f$")
plt.ylabel(r"$S_{\widetilde{K}}\left( f \right)$")
plt.xlim(-0.003, 0.03)
plt.ylim(1e-1, None)
plt.savefig("S(K)_1e-4.eps", bbox_inches="tight")
plt.show()

#
# U_fit, symbols, _, _ = create_fit_U(
#     fU, dt, U, time, "exp", td=500, shuffled=False, lam=0.1
# )
#
# plt.plot(time, U)
# plt.plot(time, U_fit, "--")
# plt.xlabel(r"$t$")
# plt.ylabel(r"$\widetilde{U}$")
# plt.show()
#
#
# f, PU_fit = signal.welch(U_fit, 1 / dt, nperseg=len(U_fit) / 4)
#
# plt.semilogy(fU, PU)
# plt.semilogy(f, PU_fit, "--")
# plt.xlabel(r"$f$")
# plt.ylabel(r"$S_{U}\left( f \right)$")
# # plt.xlim(-10, 200)
# # plt.ylim(1e-5, None)
# plt.show()

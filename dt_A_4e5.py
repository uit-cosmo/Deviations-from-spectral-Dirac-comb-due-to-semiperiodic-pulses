import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from fit_function_RB_model import create_fit_RB
import cosmoplots
from fppanalysis import corr_fun

axes_size = cosmoplots.set_rcparams_dynamo(plt.rcParams, num_cols=1, ls="thin")

ts = np.load("RB_time_series_4e5.npy")

intervals_start = [600, 3600, 6800, 9400, 12500, 15500, 18400, 21400, 24400, 27400]

dt = 0.01 / 200

amplitudes = []
duration_times = []

for i in range(len(intervals_start)):
    ts_interval = ts[intervals_start[i]: intervals_start[i] + 1600]
    ts_interval = (ts_interval - np.mean(ts_interval)) / np.std(ts_interval)
    # plt.plot(ts_interval)
    # plt.show()
    time = np.linspace(0, dt * len(ts_interval) - dt, num=len(ts_interval))
    f, Pxx = signal.welch(ts_interval, 1 / dt, nperseg=len(ts_interval) / 1)

    time_series_fit, symbols, duration_time, forcing = create_fit_RB(
        "4e5", f, dt, Pxx, ts_interval, time
    )
    amplitudes = np.append(amplitudes, forcing[forcing != 0][1:])
    arrival_times = time[forcing != 0]
    duration_times = np.append(duration_times, np.diff(arrival_times))

    print(f"Done with {i+1}/{len(intervals_start)}")

print(amplitudes)
print(duration_times)

plt.hist(duration_times, bins=32, density=True)
plt.xlabel(r"$\tau_w$")
plt.ylabel(r"$P(\tau_w)$")
plt.savefig("duration_times_4e5.eps", bbox_inches="tight")
plt.show()

plt.scatter(amplitudes, duration_times)
plt.xlabel(r'$A$')
plt.ylabel(r"$\tau_w$")
plt.savefig("amp_vs_dt_4e5.eps", bbox_inches="tight")
plt.show()

tb, R = corr_fun(amplitudes, amplitudes, dt=1)
plt.plot(tb, R, 'o--')
plt.xlim(0,20)
plt.xlabel(r'$n$')
plt.ylabel(r'$R(A_n)$')
plt.savefig("autocorr_4e5.eps", bbox_inches="tight")
plt.show()

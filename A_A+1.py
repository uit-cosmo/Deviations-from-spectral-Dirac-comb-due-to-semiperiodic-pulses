import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from fit_function_RB_model import create_fit_RB
import cosmoplots 
from fppanalysis import corr_fun

axes_size = cosmoplots.set_rcparams_dynamo(plt.rcParams, num_cols=1, ls="thin")

ts = np.load("RB_time_series_2e6.npy")

intervals_start = [10000, 18600, 21200, 28000, 30500, 33000, 37800, 40500, 45400, 52600]

dt = 0.01 / 200

amplitudes = []
duration_times = []

for i in range(len(intervals_start)):
    # + 600 in order to get the constant part
    ts_interval = ts[intervals_start[i] + 600: intervals_start[i] + 1200]
    ts_interval = (ts_interval - np.mean(ts_interval)) / np.std(ts_interval)
    # plt.plot(ts_interval)
    # plt.show()
    time = np.linspace(0, dt * len(ts_interval) - dt, num=len(ts_interval))
    f, Pxx = signal.welch(ts_interval, 1 / dt, nperseg=len(ts_interval) / 1)

    time_series_fit, symbols, duration_time, forcing = create_fit_RB(
        "2e6", f, dt, Pxx, ts_interval, time
    )

    amplitudes = np.append(amplitudes, forcing[forcing != 0])

    print(f"Done with {i+1}/{len(intervals_start)}")

print(amplitudes)

amp_0 = amplitudes[:-1]
amp_1 = amplitudes[1:]

plt.scatter(amp_0, amp_1)
plt.xlabel(r"$A_{n}$")
plt.ylabel(r"$A_{n+1}$")
plt.savefig("A_A+1_2e6.eps", bbox_inches="tight")
plt.show()

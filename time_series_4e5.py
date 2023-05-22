import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import cosmoplots
from fit_function_RB_model import create_fit_RB

axes_size = cosmoplots.set_rcparams_dynamo(plt.rcParams, num_cols=1, ls="thin")

ts = np.load("RB_time_series_4e5.npy")

intervals_start = 6800

ts_interval = ts[6800:8400]
ts_interval = (ts_interval - np.mean(ts_interval)) / np.std(ts_interval)

dt = 0.01 / 200
f, Pxx = signal.welch(ts_interval, 1 / dt, nperseg=len(ts_interval) / 1)
time = np.linspace(0, dt * len(ts_interval) - dt, num=len(ts_interval))

time_series_fit, symbols, duration_time = create_fit_RB(
    "4e5", f, dt, Pxx, ts_interval, time
)

plt.plot(time, ts_interval)
plt.plot(time, time_series_fit, "--")
plt.xlabel("t")
plt.ylabel(r"$\widetilde{n}$")
plt.show()

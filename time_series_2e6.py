import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import cosmoplots
from fit_function_RB_model import create_fit_RB

axes_size = cosmoplots.set_rcparams_dynamo(plt.rcParams, num_cols=1, ls="thin")

ts = np.load("RB_time_series_2e6.npy")

Pxx_average = [0] * 301

ts_interval = ts[31100:31700]
ts_interval = (ts_interval - np.mean(ts_interval)) / np.std(ts_interval)

dt = 0.01 / 200
f, Pxx = signal.welch(ts_interval, 1 / dt, nperseg=len(ts_interval) / 1)
time = np.linspace(0, dt * len(ts_interval) - dt, num=len(ts_interval))

time_series_fit, symbols, duration_time = create_fit_RB(
    "2e6", f, dt, Pxx, ts_interval, time
)

plt.plot(time, ts_interval)
plt.plot(time, time_series_fit, "--")
plt.xlabel(r"$t$")
plt.ylabel(r"$\widetilde{n}$")
plt.savefig("time_series_2e6.eps", bbox_inches="tight")
plt.show()

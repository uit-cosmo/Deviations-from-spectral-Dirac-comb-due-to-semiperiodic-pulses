import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import cosmoplots
from fit_function_RB_model import create_fit_RB

axes_size = cosmoplots.set_rcparams_dynamo(plt.rcParams, num_cols=1, ls="thin")

ts = np.load("RB_time_series_4e5.npy")

ts_interval = ts[6800:8400]
ts_interval = (ts_interval - np.mean(ts_interval)) / np.std(ts_interval)

dt = 0.01 / 200
f, Pxx = signal.welch(ts_interval, 1 / dt, nperseg=len(ts_interval) / 1)
time = np.linspace(0, dt * len(ts_interval) - dt, num=len(ts_interval))

time_series_fit, symbols, duration_time, forcing = create_fit_RB(
    "4e5", f, dt, Pxx, ts_interval, time
)

amplitudes = forcing[forcing != 0]
arrival_times = time[forcing != 0]
duration_times = np.diff(arrival_times)

f_forcing, Pxx_forcing = signal.welch(forcing, 1/dt, nperseg=len(forcing)/1)
plt.semilogy(f_forcing, Pxx_forcing)

plt.xlabel(r"$f$")
plt.ylabel(r"$S_{\widetilde{n}}\left( f \right)$")
plt.xlim(-50, 2000)
plt.savefig("forcing_spectra_4e5.eps", bbox_inches="tight")
plt.show()


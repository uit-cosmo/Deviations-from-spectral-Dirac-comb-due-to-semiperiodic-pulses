import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

ts = np.load("RB_time_series_4e5.npy")

intervals_start = [600, 3600, 6800, 9400, 12500, 15500, 18400, 21400, 24400, 27400]

dt = 0.01 / 200
time = np.linspace(0, dt * len(ts) - dt, num=len(ts))

Pxx_average = [0] * 801

for i in range(len(intervals_start)):
    ts_interval = ts[intervals_start[i] : intervals_start[i] + 1600]
    ts_interval = (ts_interval - np.mean(ts_interval)) / np.std(ts_interval)
    # plt.plot(ts_interval)
    # plt.show()
    f, Pxx = signal.welch(ts_interval, 1 / dt, nperseg=len(ts_interval) / 1)
    Pxx_average += Pxx

Pxx_average /= 10

plt.figure()
plt.xlabel("t")
plt.ylabel("theta")
plt.plot(time, ts)

plt.figure()
plt.semilogy(f, Pxx_average)
plt.xlabel("f")
plt.ylabel("PSD(f)")
plt.show()

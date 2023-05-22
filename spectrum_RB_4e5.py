import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from fit_function_RB_model import create_fit_RB

ts = np.load("RB_time_series_4e5.npy")

intervals_start = [600, 3600, 6800, 9400, 12500, 15500, 18400, 21400, 24400, 27400]

dt = 0.01 / 200

Pxx_average = [0] * 801
Pxx_average_fit = [0] * 801

for i in range(len(intervals_start)):
    ts_interval = ts[intervals_start[i] : intervals_start[i] + 1600]
    ts_interval = (ts_interval - np.mean(ts_interval)) / np.std(ts_interval)
    # plt.plot(ts_interval)
    # plt.show()
    time = np.linspace(0, dt * len(ts_interval) - dt, num=len(ts_interval))
    f, Pxx = signal.welch(ts_interval, 1 / dt, nperseg=len(ts_interval) / 1)

    time_series_fit, symbols, duration_time = create_fit_RB(
        "4e5", f, dt, Pxx, ts_interval, time
    )

    f_fit, Pxx_fit = signal.welch(
        time_series_fit, 1 / dt, nperseg=len(time_series_fit) / 1
    )

    plt.plot(ts_interval)
    plt.plot(time_series_fit)
    plt.show()

    Pxx_average += Pxx
    Pxx_average_fit += Pxx_fit

    print(f"Done with {i+1}/{len(intervals_start)}")

Pxx_average /= 10
Pxx_average_fit /= 10
time = np.linspace(0, dt * len(ts) - dt, num=len(ts))

plt.figure()
plt.xlabel("t")
plt.ylabel("theta")
plt.plot(time, ts)

plt.figure()
plt.semilogy(f, Pxx_average)
plt.semilogy(f_fit, Pxx_average_fit)
plt.xlabel("f")
plt.ylabel("PSD(f)")
plt.show()

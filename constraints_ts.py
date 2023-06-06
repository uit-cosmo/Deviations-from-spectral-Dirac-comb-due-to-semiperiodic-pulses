import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from fit_function_RB_model import constrained_fit_RB, return_peaks
import cosmoplots

axes_size = cosmoplots.set_rcparams_dynamo(plt.rcParams, num_cols=1, ls="thin")

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

    time_series_fit, symbols, duration_time, forcing = constrained_fit_RB(
        "4e5", f, dt, Pxx, ts_interval, time
    )

    f_fit, Pxx_fit = signal.welch(
        time_series_fit, 1 / dt, nperseg=len(time_series_fit) / 1
    )

    time_peaks, peaks = return_peaks(ts_interval, time)
    print(time_peaks, peaks)

    plt.plot(time, ts_interval)
    plt.plot(time, time_series_fit)
    plt.scatter(time_peaks, peaks)
    plt.show()

    # plt.figure()
    # plt.xlim(-100, 2000)
    # plt.ylim(10e-19, 10e-2)
    # plt.semilogy(f, Pxx)
    # plt.semilogy(f_fit, Pxx_fit, "--")
    # plt.xlabel(r"$f$")
    # plt.ylabel(r"$S_{\widetilde{n}}\left( f \right)$")
    # plt.savefig(f"spectrum_4e5_{i+1}.eps", bbox_inches="tight")
    # plt.show()

    Pxx_average += Pxx
    Pxx_average_fit += Pxx_fit

    print(f"Done with {i+1}/{len(intervals_start)}")

# Pxx_average /= 10
# Pxx_average_fit /= 10
#
# plt.figure()
# plt.xlim(-100, 2000)
# plt.ylim(10e-19, 10e-2)
# plt.semilogy(f, Pxx_average)
# plt.semilogy(f_fit, Pxx_average_fit, "--")
# plt.text(x=1500, y=1e-3, s=r"$Ra = 4\times 10^{5}$")
# plt.xlabel(r"$f$")
# plt.ylabel(r"$S_{\widetilde{n}}\left( f \right)$")
# plt.savefig("spectrum_4e5.eps", bbox_inches="tight")
# # plt.show()

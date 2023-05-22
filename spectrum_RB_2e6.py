import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

ts = np.load("RB_time_series_2e6.npy")

intervals_start = [10000, 18600, 21200, 28000, 30500, 33000, 37800, 40500, 45400, 52600]

dt = 0.01/200
time = np.linspace(0,dt*len(ts)-dt, num = len(ts))

Pxx_average = [0]*301

for i in range(len(intervals_start)):
    # + 600 in order to get the constant part 
    ts_interval = ts[intervals_start[i] + 600 :intervals_start[i]+1200]
    ts_interval = (ts_interval - np.mean(ts_interval))/np.std(ts_interval)
    # plt.plot(ts_interval)
    # plt.show()
    f, Pxx = signal.welch(ts_interval, 1/dt, nperseg=len(ts_interval)/1)
    Pxx_average += Pxx

Pxx_average /= 10

plt.figure()
plt.xlabel('t')
plt.ylabel('theta')
plt.plot(time,ts)

plt.figure()
plt.semilogy(f, Pxx_average)
plt.xlabel('f')
plt.ylabel('PSD(f)')
plt.show()

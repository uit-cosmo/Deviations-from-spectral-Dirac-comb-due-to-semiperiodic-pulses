import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

ts = np.load("RB_time_series_4e5.npy")

intervals_start = 6800

ts_interval = ts[6800:8400]
ts_interval = (ts_interval - np.mean(ts_interval))/np.std(ts_interval)

dt = 0.01/200
time = np.linspace(0,dt*len(ts_interval)-dt, num = len(ts_interval))

plt.plot(time, ts_interval)
plt.show()  

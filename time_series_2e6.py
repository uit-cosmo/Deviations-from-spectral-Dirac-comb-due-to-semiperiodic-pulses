import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

ts = np.load("RB_time_series_2e6.npy")


Pxx_average = [0]*301

ts_interval = ts[31100:31700] 
ts_interval = (ts_interval - np.mean(ts_interval))/np.std(ts_interval)

dt = 0.01/200
time = np.linspace(0,dt*len(ts_interval)-dt, num = len(ts_interval))

plt.plot(time, ts_interval)
plt.show()

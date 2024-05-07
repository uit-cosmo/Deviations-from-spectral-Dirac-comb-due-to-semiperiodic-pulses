import numpy as np
from scipy.signal import welch
from numba import njit

def gen_arrivals(tw, wrms, T):
    W = np.random.normal(tw, wrms, size = int((T*1.5)/tw))
    S = np.cumsum(W)
    S = np.sort(S)
    return S[(S>=0.25*T)&(S<=1.25*T)] - 0.25*T

@njit
def gen_signal(arrivals, T, dt):
    signal = np.zeros(int(T/dt))
    
    for i in range(arrivals.size):
        idx = int((np.round((arrivals[i])/dt)))
        signal[idx] += 1./dt # Discretized delta function
    return signal

tw = 1
Wrms = [5,1,1e-1]

T = 100_000
dt = 0.01

for i in range(len(Wrms)):
    signal = gen_signal(gen_arrivals(tw, Wrms[i], T), T, dt)
    freq, psd = welch(signal, fs=1./dt, nperseg = signal.size/(T*dt))
    psd /= 2 # divide by 2 since welch returns onesided spectrum.

    fname = 'psd_w_norm_fmax_{}_w_{:.1e}'.format(freq[-1], float(Wrms[i]))
    np.savez(fname, psd=psd, freq=freq)
    print('Wrms '+str(Wrms[i])+' done', flush=True)

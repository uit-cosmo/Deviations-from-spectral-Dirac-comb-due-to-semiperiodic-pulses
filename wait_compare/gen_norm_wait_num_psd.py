import numpy as np
from numba import njit


@njit
def gen_arrivals(tw, wrms, T):
    W = np.random.normal(tw, wrms, size=int((T * 1.5) / tw))
    S = np.cumsum(W)
    S = np.sort(S)
    return S[(S >= 0.25 * T) & (S <= 1.25 * T)] - 0.25 * T


@njit
def gen_signal(arrivals, T, dt):
    signal = np.zeros(int(T / dt))

    for i in range(arrivals.size):
        idx = int((np.round((arrivals[i]) / dt)))
        signal[idx] += 1.0 / dt  # Discretized delta function
    return signal


@njit  # ('float64[:](float64[:],float64[:])')
def gen_psd_one_signal(freq, arrivals):
    psd = np.zeros(freq.size, dtype="complex")
    for i in range(arrivals.size):
        psd += np.exp(2 * np.pi * 1.0j * freq * arrivals[i])
    return np.real(psd * np.conj(psd))


@njit  # ('float64[:](float64[:],float64, float64, float64, int64)')
def average_psd(freq, T, tw, wrms, repeat=10):
    psd = np.zeros(freq.size)
    for _ in range(repeat):
        arrivals = gen_arrivals(tw, wrms, T)
        psd += gen_psd_one_signal(freq, arrivals)
    return psd / (T * repeat)


tw = 1
Wrms = [5, 1, 1e-1]

T = 10_000
repeat = 100
dt = 0.01

for i in range(len(Wrms)):
    # signal = gen_signal(gen_arrivals(tw, Wrms[i], T), T, dt)
    # freq, psd = welch(signal, fs=1./dt, nperseg = signal.size/(T*dt))
    # psd /= 2 # divide by 2 since welch returns onesided spectrum.
    freq = np.arange(1, 501) * dt
    psd = average_psd(freq, T, tw, Wrms[i], repeat=repeat)
    fname = "psd_w_norm_fmax_{}_w_{:.1e}".format(freq[-1], float(Wrms[i]))
    np.savez(fname, psd=psd, freq=freq, T=T)
    print("Wrms " + str(Wrms[i]) + " done", flush=True)

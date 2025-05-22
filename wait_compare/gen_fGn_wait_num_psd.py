import numpy as np
from numba import njit, objmode

"""
Here, we find the PSD with waiting times based on the fGn.
"""


@njit
def gamma(k, H):
    # Correlation function of fractional Gaussian noise
    return 0.5 * (
        np.abs(k - 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k + 1) ** (2 * H)
    )


@njit
def davis_harte(N, H):
    """
    Use the Davis-Harte method to generate samples of
    G_H(k) = B_H(k+1)-B_H(k)
    where B_H(t) is the fractional Brownian motion with hurst parameter H.
    Based on DOI: 10.1017/S0269964803173081

    The returned process has zero mean and unit variance.

    Note that the Davis-Harte method may fail for large Hurst exponents (H>0.9).
    """
    a = np.zeros(2 * N)
    a[:N] = gamma(np.arange(N), H)
    a[N + 1 :] = a[1:N][::-1]

    with objmode(lam="complex128[:]"):
        lam = np.fft.ifft(a, norm="forward")
    # assert (
    #   np.real(lam) > 0
    # ).all(), "Negative elements in fourier transform. Increase number of samples."

    w = 0.5 * np.sqrt(lam / N)

    U0 = np.random.normal(loc=0.0, scale=1.0, size=2 * N)
    U1 = np.random.normal(loc=0.0, scale=1.0, size=2 * N)

    w[0] *= np.sqrt(2) * U0[0]
    w[1:N] *= U0[1:N] + 1.0j * U1[1:N]
    w[N] *= np.sqrt(2) * U0[N]
    w[N + 1 :] *= U0[N + 1 :] - 1.0j * U1[N + 1 :]

    # Need to multiply by sqrt(2) to get unit variance.
    with objmode(out="float64[:]"):
        out = np.sqrt(2) * np.real(np.fft.ifft(w, norm="forward"))[:N]
    return out


@njit
def gen_arrivals(tw, wrms, T, H):
    fGn = davis_harte(int((T * 1.5) / tw), H)
    W = wrms * fGn + tw

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
def average_psd(freq, T, tw, wrms, H, repeat=10):
    psd = np.zeros(freq.size)
    for _ in range(repeat):
        arrivals = gen_arrivals(tw, wrms, T, H)
        psd += gen_psd_one_signal(freq, arrivals)
    return psd / (T * repeat)


tw = 1
Wrms = 1
H = [0.1, 0.5, 0.9]

T = 10_000
repeat = 1000
dt = 0.001

for i in range(len(H)):
    # signal = gen_signal(gen_arrivals(tw, Wrms[i], T), T, dt)
    # freq, psd = welch(signal, fs=1./dt, nperseg = signal.size/(T*dt))
    # psd /= 2 # divide by 2 since welch returns onesided spectrum.
    freq = np.arange(1, 2001) * dt
    psd = average_psd(freq, T, tw, Wrms, H[i], repeat=repeat)
    fname = "psd_w_fGn_fmax_{}_H_{:.1e}".format(freq[-1], float(H[i]))
    np.savez(fname, psd=psd, freq=freq, T=T)
    print("Wrms " + str(H[i]) + " done", flush=True)

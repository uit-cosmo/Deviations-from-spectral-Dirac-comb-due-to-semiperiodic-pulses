import numpy as np
import scipy.signal as ssi
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, fftconvolve

def corr_fun(X, Y, dt, norm=True, biased=True, method="auto"):
    """
    Estimates the correlation function between X and Y using ssi.correlate.
    For now, we require both signals to be of equal length.

    Input:
        X: First array to be correlated. ............................. (Nx1) np.array
        Y: Second array to be correlated. ............................ (Nx1) np.array
        dt: Time step of the time series. ............................ float
        norm: Normalizes the correlation function to a maxima of 1 ... bool
        biased: Trigger estimator biasing. ........................... bool
        method: 'direct', 'fft' or 'auto'. Passed to ssi.correlate ... string

    For biased=True, the result is divided by X.size.
    For biased=False, the estimator is unbiased and returns the result
    divided by X.size-|k|, where k is the lag.
    The unbiased estimator diverges for large lags, and
    for small lags and large X.size, the difference is trivial.
    """

    assert X.size == Y.size

    if norm:
        Xn = (X - X.mean()) / X.std()
        Yn = (Y - Y.mean()) / Y.std()
    else:
        Xn = X
        Yn = Y

    R = ssi.correlate(Xn, Yn, mode="full", method=method)

    k = np.arange(-(X.size - 1), X.size)
    tb = k * dt

    if biased:
        R /= X.size
    else:
        R /= X.size - np.abs(k)

    return tb, R


def sample_asymm_laplace(alpha=1.0, kappa=0.5, size=None, seed=None):
    """
    Use:
        sample_asymm_laplace(alpha=1., kappa=0.5, size=None)
    Random samples drawn from the asymmetric Laplace distribution
    using inverse sampling. The distribution is given by
    F(A;alpha,kappa) = 1-(1-kappa)*Exp(-A/(2*alpha*(1-kappa))), A>0
                       kappa*Exp[A/(2*alpha*kappa)], A<0
    where F is the CDF of A, alpha is a scale parameter and
    kappa is the asymmetry parameter.
    Input:
        alpha: scale parameter. ......................... float, alpha>0
        kappa: shape (asymmetry) parameter .............. float, 0<=kappa<=1
        size: number of points to draw. 1 by default. ... int, size>0
        seed: specify a random seed. .................... int
    Output:
        X: Array of randomly distributed values. ........ (size,) np array
    """
    import numpy as np

    assert alpha > 0.0
    assert (kappa >= 0.0) & (kappa <= 1.0)
    if size:
        assert size > 0
    prng = np.random.RandomState(seed=seed)
    U = prng.uniform(size=size)
    X = np.zeros(size)
    X[U > kappa] = -2 * alpha * (1 - kappa) * np.log((1 - U[U > kappa]) / (1 - kappa))
    X[U < kappa] = 2 * alpha * kappa * np.log(U[U < kappa] / kappa)

    return X

def make_signal_convolve(T, amp, ta, pulse, dt):
    """
    Make a signal with prescribed amplitudes, arrival times and pulse shape by convolution.
    """
    S = np.zeros(T.size)
    for i in range(ta.size):
        S[int((ta[i]-ta[0])/dt)]+=amp[i]
    S=fftconvolve(S,pulse,mode='same')
    return S

def create_fit(dt, T, ConditionalEvents, kernrad=2**18):
    """calculates fit for K time series using already calculated 
    conditional events and assuming exponential pulses."""
    def double_exp(tkern, td, lam):
        kern = np.zeros(tkern.size)
        kern[tkern < 0] = np.exp(tkern[tkern < 0] / lam / td)
        kern[tkern >= 0] = np.exp(-tkern[tkern >= 0] / (1 - lam) / td)
        return kern

    opt, cov = curve_fit(double_exp,ConditionalEvents.time,
                                  ConditionalEvents.average/max(ConditionalEvents.average), p0=[12.,0.4])
    
    print('[td lambda]:')
    print(opt)
    print('convariance:')
    print(cov)
    time_kern = np.arange(-kernrad, kernrad + 1) * dt
    kern = double_exp(time_kern, opt[0], opt[1])

    return make_signal_convolve(T, ConditionalEvents.peaks, ConditionalEvents.arrival_times, kern, dt), (time_kern, kern)

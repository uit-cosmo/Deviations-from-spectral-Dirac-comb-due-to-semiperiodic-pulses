import numpy as np
import scipy.signal as ssi


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


def Lorentz_pulse(theta):
    return 4 * (4 + theta**2) ** (-1)


def Lorentz_PSD(theta):
    return 2 * np.pi * np.exp(-2 * np.abs(theta))


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def PSD_periodic_arrivals(omega, td, gamma, A_rms, A_mean, dt, norm=True):
    I_2 = 1 / (2 * np.pi)
    first_term = td * gamma * A_rms**2 * I_2 * Lorentz_PSD(td * omega)
    tmp = np.zeros(omega.size)
    index = np.zeros(1000)
    for n in range(1, 1000):
        index = 2 * np.pi * n * gamma
        tmp = np.where(np.abs(omega - find_nearest(omega, index)) > 0.001, tmp, 1)

    PSD = (
        2 * np.pi * td * gamma**2 * A_mean**2 * I_2 * Lorentz_PSD(td * omega) * tmp
    )

    # imitate finite amplitude in PSD  due to numerical discretization
    PSD = 2 * (first_term + PSD / dt)

    if norm:
        Phi_rms = Phi_rms_periodic_lorentz(gamma, A_rms, A_mean)
        Phi_mean = Phi_mean_periodic_lorentz(gamma, A_mean)
        PSD[0] = PSD[0] - Phi_mean**2 * 2 * np.pi
        return PSD / Phi_rms**2
    return PSD


def autocorr_periodic_arrivals(t, gamma, A_mean, A_rms, td, norm=True):
    I_2 = 1 / (2 * np.pi)
    central_peak = gamma * A_rms**2 * I_2 * Lorentz_pulse(t / td)
    oscillation = (
        gamma
        * np.pi
        * (
            1 / np.tanh(2 * np.pi * gamma - 1j * gamma * np.pi * t / td)
            + 1 / np.tanh(2 * np.pi * gamma + 1j * gamma * np.pi * t / td)
        )
    )
    R = central_peak + gamma * A_mean**2 * I_2 * oscillation.astype("float64")
    if norm:
        Phi_rms = Phi_rms_periodic_lorentz(gamma, A_rms, A_mean)
        Phi_mean = Phi_mean_periodic_lorentz(gamma, A_mean)
        return (R - Phi_mean**2) / Phi_rms**2
    return R


def Phi_rms_periodic_lorentz(gamma, A_rms, A_mean):
    I_2 = 1 / (2 * np.pi)
    return (
        gamma * A_rms**2 * I_2
        + gamma
        * A_mean**2
        * I_2
        * (2 * np.pi * gamma * (1 / np.tanh(2 * np.pi * gamma)) - gamma / I_2)
    ) ** 0.5


def Phi_mean_periodic_lorentz(gamma, A_mean):
    I_1 = 1
    return gamma * A_mean * I_1

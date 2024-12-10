import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve
import superposedpulses.forcing as frc
import closedexpressions as ce


class PeriodicAsymLapPulses(frc.ForcingGenerator):
    def __init__(self, control_parameter):
        self.control_parameter = control_parameter

    def get_forcing(self, times: np.ndarray, waiting_time: float) -> frc.Forcing:
        total_pulses = int(max(times) / waiting_time)
        arrival_time_indx = (
            np.arange(start=0, stop=99994, step=5) * 100
        )  # multiplied with inverse dt
        amplitudes = sample_asymm_laplace(
            alpha=0.5
            / np.sqrt(
                1.0 - 2.0 * self.control_parameter * (1.0 - self.control_parameter)
            ),
            kappa=self.control_parameter,
            size=total_pulses,
        )
        durations = np.ones(shape=total_pulses)
        return frc.Forcing(
            total_pulses, times[arrival_time_indx], amplitudes, durations
        )

    def set_amplitude_distribution(
        self,
        amplitude_distribution_function,
    ):
        pass

    def set_duration_distribution(self, duration_distribution_function):
        pass


def sample_asymm_laplace(alpha=1.0, kappa=0.5, size=1, seed=None):
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
        S[int((ta[i] - ta[0]) / dt)] += amp[i]
    S = fftconvolve(S, pulse, mode="same")
    return S


def create_fit(dt, T, ConditionalEvents, kernrad=2**18):
    """calculates fit for K time series using already calculated
    conditional events and assuming exponential pulses."""

    def double_exp(tkern, td, lam):
        kern = np.zeros(tkern.size)
        kern[tkern < 0] = np.exp(tkern[tkern < 0] / lam / td)
        kern[tkern >= 0] = np.exp(-tkern[tkern >= 0] / (1 - lam) / td)
        return kern

    opt, cov = curve_fit(
        double_exp,
        ConditionalEvents.time,
        ConditionalEvents.average / max(ConditionalEvents.average),
        p0=[12.0, 0.4],
    )

    print("[td lambda]:")
    print(opt)
    print("convariance:")
    print(cov)
    time_kern = np.arange(-kernrad, kernrad + 1) * dt
    kern = double_exp(time_kern, opt[0], opt[1])

    return make_signal_convolve(
        T, ConditionalEvents.peaks, ConditionalEvents.arrival_times, kern, dt
    ), (time_kern, kern, opt)


def spectrum_gauss_renewal_part(f, tw, tw_rms):
    # The part of the gaussian renewal spectrum due to the waiting times
    Omega = tw * 2 * np.pi * f
    nu = tw_rms / tw
    return np.sinh(nu**2 * Omega**2 / 2) / (
        np.cosh(nu**2 * Omega**2 / 2) - np.cos(Omega)
    )


def spectrum_gauss(f, td, lam, amean, arms, tw, tw_rms):
    # Assumes gaussian renewal arrivals and two-sided exponential pulses
    gamma = td / tw
    Omega = tw * 2 * np.pi * f
    S = (
        td * gamma * ce.psd(gamma * Omega, 1, lam) / 2
    )  # td = 1 here since the expression already contains td.
    S *= arms**2 + amean**2 * spectrum_gauss_renewal_part(f, tw, tw_rms)
    return S


def est_wait_spectrum_ECF(f, data):
    omega = 2 * np.pi * f
    # Estimate the empirical CF of data at frequencies omega.
    ECF = np.zeros(omega.size, dtype=complex)
    for i in range(omega.size):
        ECF[i] = np.mean(np.exp(1.0j * omega[i] * data))

    return np.real((1 + ECF) / (1 - ECF))


def spectrum_renewal(f, td, lam, amean, arms, tw_data):
    # Use waiting time data to estimate spectrum for renewal arrivals and two-sided exponential pulses
    gamma = td / tw_data.mean()
    omega = 2 * np.pi * f

    S = td * gamma * ce.psd(td * omega, 1, lam) / 2
    S *= arms**2 + amean**2 * est_wait_spectrum_ECF(f, tw_data)
    return S

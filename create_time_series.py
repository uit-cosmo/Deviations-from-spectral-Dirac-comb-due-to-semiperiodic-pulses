"""This module computes the time series of Lorenz system used in manuscript arXiv:2106.15904"""

import numpy as np
from scipy.integrate import solve_ivp


def lorenz_system(time, state, rho, sigma, beta):
    x, y, z = state
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z


def integrate_system(
    regime,
    params,
    init,
    dt=1e-3,
    N=2**12,
):
    """
    This function numerically integrates a system of
    ordinary differential equations given an initial value.
    """
    t_span = [0, dt * N]
    T = np.arange(0, N) * dt

    res = solve_ivp(
        lorenz_system,
        t_span,
        init[regime],
        t_eval=T,
        args=(params[regime]),
        method="Radau",
        rtol=1e-8,
        atol=1e-8,
    )

    return T, res


def main():
    """Computes time series for regimes used in manuscript arXiv:2106.15904"""

    regimes = ["rho=28", "rho=220", "rho=350"]

    params = {
        "rho=28": (28.0, 10.0, 8.0 / 3.0),
        "rho=220": (220.0, 10.0, 8.0 / 3.0),
        "rho=350": (350.0, 10, 8.0 / 3.0),
    }

    init = {
        "rho=28": [-4, -7, 25],
        "rho=220": [45, 60, 275],
        "rho=350": [-8, -46, 284],
    }

    for regime in regimes:
        dname = "lorenz_" + regime + ".npz"
        print("starting " + dname, flush=True)
        T, res = integrate_system(regime, params, init, N=2**21)
        np.savez(dname, tb=T, res=res["y"])
        print(dname + " done", flush=True)


if __name__ == "__main__":
    main()

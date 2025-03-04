import numpy as np
import mpmath as mm

tw = mm.mpf("1")
Wrms_list = ["5", "1", "1e-1"]

Wrms = [mm.mpf(w) for w in Wrms_list]
A = [(tw / w) ** 2 + 2 for w in Wrms]
B = [(a - 1) * tw for a in A]

fs = mm.mpf("100")
freq_ind = mm.arange(1, 10 * fs + 1)
omega = [2 * mm.pi * f / fs for f in freq_ind]


def cf_invgamma(omega, a, b):
    x = mm.sqrt(-1.0j * b * omega)
    return 2 * x**a * mm.besselk(a, 2 * x) / mm.gamma(a)


for i in range(len(Wrms_list)):
    fname = "cf_invgamma_fmax_{}_w_{:.1e}".format(
        freq_ind[-1] / fs, float(Wrms_list[i])
    )
    C = np.array([cf_invgamma(o, A[i], B[i]) for o in omega], dtype=complex)
    np.savez(fname, cf=C, omega=np.array(omega, dtype="float"))
    print("Wrms " + Wrms_list[i] + " done", flush=True)

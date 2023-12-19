import numpy as np
from fppanalysis import cond_av
from scipy import signal
import matplotlib.pyplot as plt
from support_functions import create_fit
import cosmoplots


axes_size = cosmoplots.set_rcparams_dynamo(plt.rcParams, num_cols=1, ls="thin")

# Match \mathcal{K} in text
plt.rcParams["font.family"] = "serif"

mu_list = ['1.6e-3','1e-4']
wait_min = {mu_list[0]:50,
            mu_list[1]:200}
ts_lim = {mu_list[0]:[20000,22000, None, None],
        mu_list[1]:[70000,72000, None, 15]}
spectra_lim = {mu_list[0]:[0, 0.1, 1e-1, None],
               mu_list[1]:[0, 3e-2, 1e0, None]}

def plot_RB(mu,fit=False):
    K = np.load("./RB_data/K_"+mu+"_data.npy")
    time = np.load("./RB_data/time_"+mu+"_data.npy")

    dt = time[1] - time[0]
    
    K = (K - np.mean(K)) / np.std(K)
    fK, PK = signal.welch(K, 1 / dt, nperseg=len(K) / 4)
    
    plt.figure('K_ts'+mu)
    plt.plot(time, K)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\widetilde{\mathcal{K}}$")
    plt.xlim(ts_lim[mu][:2])
    plt.ylim(ts_lim[mu][2:])
    
    plt.figure('S_K'+mu)
    ax = plt.gca()
    cosmoplots.change_log_axis_base(ax, "y")
    ax.plot(fK, PK)
    plt.xlabel(r"$f$")
    plt.ylabel(r"$S_{\widetilde{\mathcal{K}}}\left( f \right)$")
    plt.xlim(spectra_lim[mu][:2])
    plt.ylim(spectra_lim[mu][2:])

    if fit:
        wait = cond_av(K, time, smin=1, window=True, delta=wait_min[mu])[-1]
        K_fit = create_fit(dt, K, time, td=8, lam=0.4, distance=50, kerntype='double_exp')
        f_fit, PK_fit = signal.welch(K_fit, 1 / dt, nperseg=len(K_fit) / 4)

        wait = wait[wait > wait_min[mu]]
        plt.figure('wait_hist'+mu)
        plt.hist(wait / np.mean(wait), 32, density=True)
        plt.xlabel(r"$\tau_w/\langle\tau_w\rangle$")
        plt.ylabel(r"$P(\tau_w/\langle\tau_w\rangle)$")
        plt.savefig("P(tau)_"+mu+".eps", bbox_inches="tight")
        plt.close('wait_hist'+mu)

        plt.figure('K_ts'+mu)
        plt.plot(time, K_fit, "--")

        plt.figure('S_K'+mu)
        plt.semilogy(f_fit, PK_fit, "--")

        plt.figure('K_ts'+mu)
        plt.savefig("K_"+mu+"fit.eps", bbox_inches="tight")
        
        plt.figure('S_K'+mu)
        plt.savefig("S(K)_"+mu+"fit.eps", bbox_inches="tight")

    else:
        plt.figure('K_ts'+mu)
        plt.savefig("K_"+mu+".eps", bbox_inches="tight")
        
        plt.figure('S_K'+mu)
        plt.savefig("S(K)_"+mu+".eps", bbox_inches="tight")

    plt.close('K_ts'+mu)
    plt.close('S_K'+mu)

for mu in mu_list:
    for fit in [False,True]:
        plot_RB(mu,fit)

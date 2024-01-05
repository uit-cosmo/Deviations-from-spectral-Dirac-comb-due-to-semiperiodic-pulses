import numpy as np
from plasmapy.analysis.time_series.conditional_averaging import ConditionalEvents
from scipy import signal
import matplotlib.pyplot as plt
from support_functions import create_fit
from fppanalysis import get_hist,distribution
import cosmoplots


axes_size = cosmoplots.set_rcparams_dynamo(plt.rcParams, num_cols=1, ls="thin")

# Match \mathcal{K} in text
plt.rcParams["font.family"] = "serif"

mu_list = ['1.6e-3','1e-4']
mu_label = {mu_list[0]:r"$1.6\times 10^{-3}$",
            mu_list[1]:r"$10^{-4}$"}
mu_col = {mu_list[0]:"C1",
            mu_list[1]:"C2"}
wait_min = {mu_list[0]:60,
            mu_list[1]:300}
ts_lim = {mu_list[0]:[20000,22000, None, None],
        mu_list[1]:[70000,72000, None, 15]}
spectra_lim = {mu_list[0]:[0, 0.1, 1e-1, None],
               mu_list[1]:[0, 3e-2, 1e0, None]}

def plot_RB(mu,fit=False):
    K = np.load("./RB_data/K_"+mu+"_data.npy")
    time = np.load("./RB_data/time_"+mu+"_data.npy")

    dt = time[1] - time[0]

    
    nK = (K - np.mean(K)) / np.std(K)
    fK, PK = signal.welch(nK, 1 / dt, nperseg=len(nK) / 4)
   
    plt.figure('K_ts'+mu)
    plt.plot(time, nK)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\widetilde{\mathcal{K}}$")
    plt.xlim(ts_lim[mu][:2])
    plt.ylim(ts_lim[mu][2:])
    
    plt.figure('S_K'+mu)
    ax = plt.gca()
    cosmoplots.change_log_axis_base(ax, "y")
    ax.plot(fK[1:], PK[1:])
    plt.xlabel(r"$f$")
    plt.ylabel(r"$S_{\widetilde{\mathcal{K}}}\left( f \right)$")
    plt.xlim(spectra_lim[mu][:2])
    plt.ylim(spectra_lim[mu][2:])

    if fit:
        CoEv = ConditionalEvents(signal=K, time = time, lower_threshold=K.mean()+K.std(), distance = wait_min[mu], remove_non_max_peaks=True)
        print(CoEv.arrival_times[0])
        print(CoEv.arrival_times[-1])
        
        print('mu={}'.format(mu))
        print('<K>={}'.format(np.mean(K)))
        print('K_rms={}'.format(np.std(K)))
        
        K_fit, pulse = create_fit(dt, time, CoEv) # pulse contains (time_kern, kern)
        nK_fit = (K_fit-np.mean(K_fit))/np.std(K_fit)

        print('<K_fit>={}'.format(np.mean(K_fit)))
        print('K_fit_rms={}'.format(np.std(K_fit)),flush=True)

        plt.figure('Kav'+mu)
        plt.plot(CoEv.time, CoEv.average/max(CoEv.average), c=mu_col[mu])
        #plt.plot(CoEv.time, CoEv.variance)
        plt.plot(pulse[0], pulse[1], 'k--')
        plt.xlim([-50,50])
        plt.xlabel(r"$t$")
        plt.ylabel(r"$K_{av}$")
        plt.savefig('Kav_'+mu+'.eps')
        plt.close('Kav_'+mu+'.eps')

        f_fit, PK_fit = signal.welch(nK_fit, 1 / dt, nperseg=int(len(nK_fit) / 4))

        plt.figure('wait_hist')
        pdf, _, x = distribution(CoEv.waiting_times / np.mean(CoEv.waiting_times), 32, kernel=True)
        plt.plot(x, pdf, label=r'$\mu=$'+mu_label[mu], c=mu_col[mu])
        plt.xlabel(r"$\tau_\mathrm{w}/\langle\tau_\mathrm{w}\rangle$")
        plt.ylabel(r"$P(\tau_\mathrm{w}/\langle\tau_\mathrm{w}\rangle)$")

        plt.figure('amp_hist')
        pdf, _, x = distribution(CoEv.peaks / np.mean(CoEv.peaks), 32, kernel=True)
        plt.plot(x, pdf, label=r'$\mu=$'+mu_label[mu], c=mu_col[mu])
        plt.xlabel(r"$A/\langle A\rangle$")
        plt.ylabel(r"$P(A/\langle A\rangle)$")

        plt.figure('K_ts'+mu)
        plt.plot(time+(CoEv.arrival_times[0]-time[0]), nK_fit, "--", c=mu_col[mu])
        
        plt.figure('S_K'+mu)
        plt.semilogy(f_fit, PK_fit, "--", c=mu_col[mu])

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

plt.figure("wait_hist")
plt.legend()
plt.savefig("Ptau.eps", bbox_inches="tight")
plt.close('wait_hist')

plt.figure("amp_hist")
plt.legend()
plt.savefig("PA.eps", bbox_inches="tight")
plt.close('amp_hist')

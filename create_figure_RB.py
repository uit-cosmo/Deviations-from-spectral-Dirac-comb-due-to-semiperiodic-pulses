import numpy as np
from plasmapy.analysis.time_series.conditional_averaging import ConditionalEvents
from scipy import signal
import support_functions as sf
from fppanalysis import get_hist,distribution
import cosmoplots
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use("cosmoplots.default")

# Match \mathcal{K} in text
plt.rcParams["font.family"] = "serif"

mu_list = ['1.6e-3','1e-4']

class MuOpts:
    def __init__(self, mu):
        if mu == mu_list[0]:
            self.mu = mu_list[0]
            self.savename = '1_6e-3'
            self.label =r"$1.6\times 10^{-3}$"
            self.color = "C1"
            self.wait_min = 60
            self.ts_lim = [20000,22000, None, None]
            self.spectra_lim = [0, 0.1, 1e-6, None]
        elif mu == mu_list[1]:
            self.mu = mu_list[1]
            self.savename = '1e-4'
            self.label =r"$10^{-4}$"
            self.color = "C2"
            self.wait_min = 300
            self.ts_lim =[70000,72000, -1, 14] 
            self.spectra_lim =[0, 3e-2, 1e-6, None] 

def plot_RB(Mu,fit=False):
    K = np.load("./RB_data/K_"+Mu.mu+"_data.npy")
    time = np.load("./RB_data/time_"+Mu.mu+"_data.npy")

    dt = time[1] - time[0]

    
    nK = (K - np.mean(K)) / np.std(K)
    fK, PK = signal.welch(nK, 1 / dt, nperseg=len(nK) / 4)
   
    plt.figure('K_ts'+Mu.savename)
    plt.plot(time, nK)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\widetilde{\mathcal{K}}$")
    plt.axis(Mu.ts_lim)
    if Mu.mu == mu_list[1]:
        plt.yticks(range(0,15,3))
    
    plt.figure('S_K'+Mu.savename)
    ax = plt.gca()
    cosmoplots.change_log_axis_base(ax, "y")
    ax.plot(fK[1:], PK[1:]*K.std()**2)
    plt.ylabel(r"$\mathcal{K}_\mathrm{rms}^2 S_{\widetilde{\mathcal{K}}}\left( f \right)$")
    plt.xlabel(r"$f$")
    plt.axis(Mu.spectra_lim)

    if fit:
        CoEv = ConditionalEvents(signal=K, time = time, lower_threshold=K.mean()+K.std(), distance = Mu.wait_min, remove_non_max_peaks=True)
        
        fitfile = open('fitdata_'+Mu.savename+'.txt','w')
        fitfile.write('<K>={}, K_rms={}\n'.format(np.mean(K),np.std(K)))
        
        K_fit, pulse = sf.create_fit(dt, time, CoEv) # pulse contains (time_kern, kern, (td, lam))
        nK_fit = (K_fit-np.mean(K_fit))/np.std(K_fit)
        
        fitfile.write('<K_fit>={}, K_fit_rms={}\n'.format(np.mean(K_fit),np.std(K_fit)))
        fitfile.write('td={}, lam={}\n <tw>={},tw_rms={}\n <A>={}, A_rms={}'.format(pulse[2][0],pulse[2][1],
                                                                                    CoEv.waiting_times.mean(),CoEv.waiting_times.std(),
                                                                                    CoEv.peaks.mean(),CoEv.peaks.std()))
        fitfile.close()

        plt.figure('Kav'+Mu.savename)
        plt.plot(CoEv.time, CoEv.average/max(CoEv.average), c=Mu.color)
        #plt.plot(CoEv.time, CoEv.variance)
        plt.plot(pulse[0], pulse[1], 'k--')
        plt.xlim([-50,50])
        plt.xlabel(r"$t$")
        plt.ylabel(r"$\langle \mathcal{K}(t-s) | \mathcal{K}(s)=\mathcal{K}_\mathrm{max}\rangle$")
        plt.savefig('Kav_'+Mu.savename+'.eps')
        plt.close('Kav_'+Mu.savename+'.eps')

        f_fit, PK_fit = signal.welch(nK_fit, 1 / dt, nperseg=int(len(nK_fit) / 4))

        plt.figure('wait_hist')
        pdf, _, x = distribution(CoEv.waiting_times / np.mean(CoEv.waiting_times), 32, kernel=True)
        plt.plot(x, pdf, label=r'$\mu=$'+Mu.label, c=Mu.color)
        plt.xlabel(r"$\tau_\mathrm{w}/\langle\tau_\mathrm{w}\rangle$")
        plt.ylabel(r"$P(\tau_\mathrm{w}/\langle\tau_\mathrm{w}\rangle)$")

        plt.figure('amp_hist')
        pdf, _, x = distribution(CoEv.peaks / np.mean(CoEv.peaks), 32, kernel=True)
        plt.plot(x, pdf, label=r'$\mu=$'+Mu.label, c=Mu.color)
        plt.xlabel(r"$A/\langle A\rangle$")
        plt.ylabel(r"$P(A/\langle A\rangle)$")

        plt.figure('K_ts'+Mu.savename)
        plt.plot(time+(CoEv.arrival_times[0]-time[0]), nK_fit, "--", c=Mu.color)
        
        plt.figure('S_K'+Mu.savename)
        ax.plot(f_fit, PK_fit*K_fit.std()**2, "--", c=Mu.color)
        ax.plot(f_fit, sf.spectrum_renewal(f_fit, pulse[2][0], pulse[2][1],
                                                CoEv.peaks.mean(), CoEv.peaks.std(),
                                                CoEv.waiting_times),
                     'k--')

        plt.figure('K_ts'+Mu.savename)
        plt.savefig("K_"+Mu.savename+"fit.eps")
        
        plt.figure('S_K'+Mu.savename)
        plt.savefig("S_K_"+Mu.savename+"fit.eps")

        plt.figure('Compare_renewal_gauss'+Mu.savename)
        ax2 = plt.gca()
        cosmoplots.change_log_axis_base(ax2, "y")
        ax2.plot(f_fit, sf.est_wait_spectrum_ECF(f_fit, CoEv.waiting_times),
                     'k--', label=r'$\mathrm{ECF}$')
        ax2.plot(f_fit, sf.spectrum_gauss_renewal_part(f_fit, 
                                                           CoEv.waiting_times.mean(), 
                                                           CoEv.waiting_times.std()),
                     c=Mu.color, label=r'$\mathrm{Gauss}$')
        plt.ylabel(r"$\mathcal{K}_\mathrm{rms}^2 S_{\widetilde{\mathcal{K}}}\left( f \right)$")
        plt.xlabel(r"$f$")
        plt.legend()
        plt.xlim(Mu.spectra_lim[:2])
        plt.savefig('S_compare_renewal_gauss_'+Mu.savename+'.eps')
        plt.close('Compare_renewal_gauss'+Mu.savename)

    else:
        plt.figure('K_ts'+Mu.savename)
        plt.savefig("K_"+Mu.savename+".eps")
        
        plt.figure('S_K'+Mu.savename)
        plt.savefig("S_K_"+Mu.savename+".eps")

    plt.close('K_ts'+Mu.savename)
    plt.close('S_K'+Mu.savename)

for mu in mu_list:
    for fit in [False,True]:
        plot_RB(MuOpts(mu),fit)


plt.figure("wait_hist")
plt.legend()
plt.savefig("Ptau.eps")
plt.close('wait_hist')

plt.figure("amp_hist")
plt.legend()
plt.savefig("PA.eps")
plt.close('amp_hist')

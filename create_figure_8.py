import numpy as np
import matplotlib.pyplot as plt
import cosmoplots
plt.style.use("cosmoplots.default")

def cf_norm(omega, tw, wrms):
    return np.exp(1.j*tw*omega - 0.5*wrms**2*omega**2)

def cf_gamma(omega, tw, wrms):
    return (1-1.j*omega*wrms**2/tw)**(-(tw/wrms)**2)

def cf_unif(omega, tw, wrms):
    cf = np.exp(1.j*omega*tw)*np.sin(np.sqrt(3)*wrms*omega)/(np.sqrt(3)*wrms*omega)
    cf[omega==0] = 1
    return cf

def spectrum_waiting_time_part(omega, charfun):
    cf = charfun(omega)
    return np.real((1+cf)/(1-cf))

def psd_norm_num(wrms):
    dname = 'psd_w_norm_fmax_5.0_w_{:.1e}.npz'.format(wrms)
    D = np.load(dname)
    return D['freq'], D['psd']

def psd_invgamma(wrms):
    dname = 'cf_invgamma_fmax_10.0_w_{:.1e}.npz'.format(wrms)
    D = np.load(dname)
    return D['omega']/(2*np.pi), spectrum_waiting_time_part(D['omega'], lambda o: D['cf'])


linestyle = ['-','--', '-.', ':']
labels = [r"$\mathrm{Normal\,num.}$",r"$\mathrm{Inverse\, Gamma}$",r"$\mathrm{Normal\,an.}$", r"$\mathrm{Gamma}$",]
wrms_label = [r"$1/10$", r"$1$",r"$5$"]
Wrms = [1e-1, 1, 5] 
subfiglabel = [r"$\mathrm{(a)}$",r"$\mathrm{(b)}$",r"$\mathrm{(c)}$"]

tw = 1.

dt = 0.01
f = np.arange(1,int(10/dt)+1)*dt
omega = 2*np.pi*f

rows = 3
columns = 1
fig, ax = cosmoplots.figure_multiple_rows_columns(rows, columns)

ax[2].set_xscale('log')
for row in range(rows*columns):
    for psd, ls, lab in zip([psd_norm_num, psd_invgamma], linestyle[:2], labels[:2]):
        if row == 1:
            ax[row].semilogy(*psd(Wrms[row]), ls=ls, label=lab)
        else:
            ax[row].semilogy(*psd(Wrms[row]), ls=ls)

    for cf, ls, lab in zip([cf_norm, cf_gamma], linestyle[2:], labels[2:]):
        if row == 1:
            ax[row].semilogy(f, spectrum_waiting_time_part(omega, lambda o: cf(o, 1, Wrms[row])), ls = ls, label=lab)
        else:
            ax[row].semilogy(f, spectrum_waiting_time_part(omega, lambda o: cf(o, 1, Wrms[row])), ls = ls)


    ax[row].set_ylabel(r"$S_\Phi(\omega)/\left[ \tau_\mathrm{d} \gamma \langle A \rangle^2 I_2 \varrho(\tau_\mathrm{d} \omega) \right]$")
    ax[row].set_xlabel(r"$\langle w \rangle f$")
    if row == 2:
        ax[2].set_xlim(1e-2,5)
    else:
        ax[row].set_xlim(0,5)
    ax[row].set_ylim(1e-1,2e1)
    ax[row].text(0.8, 0.1,r"$w_\mathrm{rms}/\langle w \rangle=\,$" + wrms_label[row], horizontalalignment='center', verticalalignment='center', transform=ax[row].transAxes)
    ax[row].text(-0.15, 1, subfiglabel[row], horizontalalignment='center', verticalalignment='center', transform=ax[row].transAxes)
    cosmoplots.change_log_axis_base(ax[row], base=10)
#ax[1].semilogy(f, spectrum_waiting_time_part(omega, lambda o: cf_ray(o, 1)), ls = (0, (3, 1, 1, 1, 1, 1)), label=r"$\mathrm{Rayleigh}$")
ax[1].legend()


fig.savefig("PSD_compare_wdist.eps")

Wrms = [1e-3, 1e-2]
for wrms in Wrms:
    print('Wrms:{}'.format(wrms))
    for cf, label in zip([cf_gamma, cf_unif],['gamma', 'unif']):
        diff = np.max(np.abs(spectrum_waiting_time_part(omega, lambda o: cf_norm(o, 1, wrms))-spectrum_waiting_time_part(omega, lambda o: cf(o, 1, wrms)))/spectrum_waiting_time_part(omega, lambda o: cf_norm(o, 1, wrms)))
        print('Max relative difference normal ' + label + ' ' + str(diff) )

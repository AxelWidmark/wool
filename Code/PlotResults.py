from LoadStream import load_stream
from TfFunctions import tffunc
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import corner
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d
pi = np.pi


mpl.rcParams['font.serif'] = "Times"
mpl.rcParams["mathtext.fontset"] = "dejavuserif"
mpl.rcParams['font.family'] = "serif"
mpl.rcParams.update({'font.size': 12})


# this function produces Fig. 3 of the paper
def make_err_plot():
    Gmags = np.arange(12,21)
    uncertainties = 1e-3*np.array([ [6.7, 10.3, 16.5, 26.6, 43.6, 74.5, 137, 280, 627], [3.5, 5.4, 8.7, 14.0, 22.9, 39.2, 72.3, 147, 330] ])
    log10_par_err_of_G = interp1d(Gmags, np.log10(uncertainties[0]), bounds_error=False, fill_value=(np.log10(uncertainties[0][0]),np.log10(uncertainties[0][-1])))
    log10_mu_err_of_G = interp1d(Gmags, np.log10(uncertainties[1]), bounds_error=False, fill_value=(np.log10(uncertainties[1][0]),np.log10(uncertainties[1][-1])))
    f, ax = plt.subplots(2, 1, figsize=(4.8,6))
    for i in range(2):
        ax[i].tick_params(direction='in', which='both')
        ax[i].xaxis.set_ticks_position('both')
        ax[i].yaxis.set_ticks_position('both')
        ax[i].set_xlim([6.5,20.5])
    Gvec = np.linspace(7.,20.,1000)
    ax[1].semilogy(Gmags, uncertainties[0], 'ko')
    ax[1].semilogy(Gmags, uncertainties[1], 'ko')
    ax[1].semilogy(Gvec, 10.**log10_par_err_of_G(Gvec), 'k-', label=r'$\hat{\sigma}_{\varpi}$ (mas)')
    ax[1].semilogy(Gvec, 10.**log10_mu_err_of_G(Gvec), 'k--', label=r'$\hat{\sigma}_{\mu}$ (mas/yr)')
    isochrones_appGs = np.load('../GeneratedStreams/mock_isochrone_Gmags.npy')
    isochrones_appGs += 10. # at 1 kpc distance
    print(len(isochrones_appGs))
    isochrones_appGs = isochrones_appGs[isochrones_appGs<20.][0:4000]
    print(len(isochrones_appGs))
    print(np.sum(isochrones_appGs<14.)/len(isochrones_appGs))
    print(min(isochrones_appGs))
    ax[0].hist(isochrones_appGs, np.linspace(7.,20.,14), label=r'Stellar pop. at 1 kpc')
    ax[0].plot([14.,14.], [0.,750.], 'r--', label=r'$v_{\mathrm{RV}}$ limit')
    ax[0].set_ylim([0.,750.])
    ax[1].set_ylim([2e-3,1.])
    ax[1].set_yticks([1e-2,1e-1])
    ax[0].set_yticks([200,400,600])
    ax[1].set_xlabel(r'$m_G$')
    ax[1].set_ylabel(r'Uncertainty')
    ax[0].set_ylabel(r'Number count')
    handles, labels = ax[0].get_legend_handles_labels()
    ax[0].legend(handles[::-1], labels[::-1])
    ax[1].legend()
    ax[0].set_xticklabels([])
    plt.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig('../Figs/errors.png', bbox_inches='tight', dpi=600)
    plt.show()
    return 


# this function produces the subpanels of Fig. 6 of the paper
def make_corr_plot(res_path, save_path=''):
    npzfile = np.load(res_path)
    obs_list = npzfile['obs_list']
    samples = npzfile['samples']
    param_samples = npzfile['param_samples']
    df = pd.DataFrame(param_samples, columns = ['Rho', 'Dist_0', 'Fx', 'Vx_0', 'Vy_0', 'Vz_0', 'Ang_0', 'Sigma_vx', 'Sigma_vy', 'Sigma_vz', 'Sigma_x', 'Sigma_y', 'Z_sun', 'Vx_sun', 'Vy_sun', 'Vz_sun'])
    corrmat = np.array( df[['Rho', 'Fx', 'Vx_0', 'Vy_0', 'Vz_0', 'Sigma_vx', 'Sigma_vy', 'Sigma_vz', 'Sigma_x', 'Sigma_y', 'Dist_0', 'Ang_0', 'Z_sun', 'Vx_sun', 'Vy_sun', 'Vz_sun']].corr() )
    print(corrmat)
    names = [r'$\Sigma_{240}$', r'$F_{\tilde{X}}$', r'$\tilde{U}_0$', r'$\tilde{V}_0$', r'$\tilde{W}_0$', r'$\sigma_{\tilde{U}}$', r'$\sigma_{\tilde{V}}$', r'$\sigma_{\tilde{W}}$', r'$\sigma_{\tilde{X}}$', r'$\sigma_{\tilde{Y}}$', r'$d_0$', r'$l_0$', r'$Z_\odot$', r'$U_\odot$', r'$V_\odot$', r'$W_\odot$']
    fig, ax = plt.subplots(figsize=(5,5)) 
    sns.heatmap(corrmat, vmin=-1., vmax=1., square=True, center=0, xticklabels=names, yticklabels=names, cmap=sns.diverging_palette(20, 220, n=200), ax=ax, cbar=True, cbar_kws={"shrink": .81, "ticks": [-1.,-0.5,0.,0.5,1.], 'label': 'Correlation'})
    if save_path!='':
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
    plt.show()
    return


# this function produces Fig. 4 of the paper
def make_posterior_plot(save_path=''):
    true_surf_dens_value = 37.067
    surf_dens_over_rho0 = 382.767
    effective_dens_derivRterms = -0.00222176
    stellar_start = 0
    distance = '1kpc'
    dispersions = '20-1000'
    paths = ['../Results/HMCchain_'+dispersions+'_slow_300_'+str(stellar_start)+'_'+distance+'_90deg_updated.npz', \
             '../Results/HMCchain_'+dispersions+'_corotB_300_'+str(stellar_start)+'_'+distance+'_180deg_updated.npz', \
             '../Results/HMCchain_'+dispersions+'_corotC_300_'+str(stellar_start)+'_'+distance+'_180deg_updated.npz', \
             '../Results/HMCchain_'+dispersions+'_inclined_300_'+str(stellar_start)+'_'+distance+'_90deg_updated.npz']
    for j in range(0):
        print(j)
        for i in range(len(paths)):
            npzfile = np.load(paths[i])
            param_samples = npzfile['param_samples'][0:-1:100]
            pss = param_samples[:,j]
            #plt.plot((pss-np.median(pss))/np.cov(pss)**0.5 + 3*i)
            plt.plot(pss[0:-1:100])
        plt.show()
    f, ax = plt.subplots(1, 1, figsize=(5,5))
    from gala.potential import MilkyWayPotential
    import astropy.units as u
    pot=MilkyWayPotential()
    true_value = (pot.density([8.172,0.,0.])/1e9/(u.solMass/u.kpc**3))[0]
    #ax.plot([true_value,true_value], [0.,2.5e4], 'k:')
    colors = ['C0', 'C1', 'C2', 'C3']
    labels = ['S1', 'S2', 'S3', 'S4']
    #labels = ['A', 'B', 'C', r'C, $v_{\mathrm{RV}}$ incl.', 'D', 'E']
    lss = ['-', '--', '-.', ':']
    from sklearn.mixture import GaussianMixture as GMM
    max_y = 0.
    for i in range(len(paths)):
        npzfile = np.load(paths[i])
        param_samples = npzfile['param_samples'][0:-1:100]
        print(np.shape(param_samples))
        print('Sigma_rel:',np.cov(param_samples[:,0])**0.5/true_value)
        print(true_value)
        print('16ths, 84ths:',np.percentile(param_samples[:,0],16)+effective_dens_derivRterms, np.percentile(param_samples[:,0],84)+effective_dens_derivRterms)
        #ax.hist(param_samples[:,0]+effective_dens_derivRterms, 41, histtype='step', linestyle=lss[i], linewidth=2, color=colors[i])
        #ax.plot([], [], linestyle=lss[i], linewidth=2, color=colors[i], label=labels[i])
        gmm = GMM(n_components=3,max_iter=10000,tol=0.0001,n_init=5)
        gmm.fit([[ps+effective_dens_derivRterms] for ps in param_samples[:,0]],GMM)
        f = lambda x:np.sum( [gmm.weights_[i] * np.exp( -(x-gmm.means_[i][0])**2. / (2.*gmm.covariances_[i][0,0]) ) / np.sqrt(2.*pi*gmm.covariances_[i][0,0]) for i in range(3)] )
        #ax.plot(np.linspace(0.06,0.13,1000), [f(x) for x in np.linspace(0.06,0.13,1000)], linestyle=lss[i], linewidth=2, color=colors[i], label=labels[i])
        xs = np.linspace(20., 60., 1000)
        ys = [f(x/surf_dens_over_rho0) for x in xs]
        max_y = max(max_y, max(ys))
        ax.plot(xs, ys, linestyle=lss[i], linewidth=2, color=colors[i], label=labels[i])
        print()
    ax.plot([true_surf_dens_value,true_surf_dens_value], [0.,1.1*max_y], 'k--')
    ax.tick_params(direction='in')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    #ax.set_xlim([0.065,0.12])
    ax.set_xlim([25.,55.])
    ax.set_ylim([0.,1.1*max_y])
    ax.set_yticks([])
    #ax.set_xlabel(r'$\rho_0$ (M$_{\odot}/$pc$^{3}$)')
    ax.set_xlabel(r'$\Sigma_{240}$ (M$_{\odot}/$pc$^{2}$)')
    ax.set_ylabel('Marginalised posterior density')
    plt.rcParams.update({'legend.handlelength': 2.8})
    ax.legend(loc=0)
    #ax.text(33.8,1950,r'$36.9 \pm 0.9$ M$_{\odot}/$pc$^{2}$', fontsize=12)
    if save_path!='':
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
    plt.show()
    return 


# this function produces Fig. 5 of the paper
def make_data_plot(res_path, save_path=''):
    npzfile = np.load(res_path)
    xyzs = npzfile['xyzs']
    obs_list = npzfile['obs_list']
    #print(np.sum(obs_list[:,9]<14.))
    #return
    for i in range(6,9):
        print(np.median(obs_list[:,i]))
    samples = npzfile['samples']
    param_samples = npzfile['param_samples']
    f = tffunc(obs_list, 2.141e-13, 0.015, [-11.1,245.8,7.2])
    ideal_trajec = f.get_ideal_trajectory(np.median(param_samples, axis=0), zlims=[min(xyzs[:,2]),max(xyzs[:,2])])
    #ideal_trajecs = [f.get_ideal_trajectory(ps, zlims=[min(xyzs[:,2]),max(xyzs[:,2])]) for ps in param_samples[0:-1:int(len(param_samples)/10)]]
    labels_vec = [r'$\varpi$ (mas)', r'$l$ (rad)', r'$b$ (rad)', r'$\mu_l$  (mas/yr)', r'$\mu_b$  (mas/yr)', r'$v_{\mathrm{RV}}$  (km/s)']
    fig, axs = plt.subplots(6, 6, figsize=(11.,11.))
    for i in range(6):
        axs[5][i].set_xlabel(labels_vec[i])
        if i>0:
            axs[i][0].set_ylabel(labels_vec[i])
        for j in range(6):
            axs[i][j].tick_params(direction='in')
            axs[i][j].xaxis.set_ticks_position('both')
            axs[i][j].yaxis.set_ticks_position('both')
            axs[i][j].xaxis.set_major_locator(plt.MaxNLocator(2))
            axs[i][j].yaxis.set_major_locator(plt.MaxNLocator(2))
            if i>0 or i==j:
                axs[j][i].set_yticklabels([])
            if j<5:
                axs[j][i].set_xticklabels([])
        for j in range(i+1,6):
            axs[i][j].set_visible(False)
    for i in range(6):
        if i!=5:
            axs[i][i].hist(obs_list[:,i], 10, histtype='step', color='k')
        else:
            subset = obs_list[np.where(obs_list[:,9]<14.),i][0]
            print(subset)
            axs[i][i].hist(subset, 10, histtype='step', color='k')
        axs[i][i].set_yticks([])
        for j in range(i+1,6):
            if j!=5:
                axs[j][i].plot(obs_list[:,i], obs_list[:,j], 'k.', alpha=0.3)
            else:
                vRV_xlims = axs[4][i].get_xlim()
                axs[j][i].plot(obs_list[np.where(obs_list[:,9]<14.),i][0], obs_list[np.where(obs_list[:,9]<14.),j][0], 'k.', alpha=0.3)
                axs[j][i].set_xlim(vRV_xlims)
            axs[j][i].plot(ideal_trajec[:,i], ideal_trajec[:,j], 'r')
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    if save_path!='':
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
    plt.show()
    return


# this function produces the figures in Appendix B of the paper
def make_stream_coord_plot(path, save_path=''):
    npzfile = np.load(path)
    xyzs = npzfile['xyzs']
    xyzs = xyzs[0:300*int(len(xyzs)/300):int(len(xyzs)/300)]
    obs_list = npzfile['obs_list']
    xyzs[:,0] -= np.mean(xyzs[:,0])
    xyzs[:,1] -= np.median(xyzs[:,1])
    print(len(xyzs))
    labels_vec = [r'$\tilde{X}$ (kpc)', r'$\tilde{Y}$ (kpc)', r'$\tilde{Z}$ (kpc)', r'$\tilde{U}$ (km/s)', r'$\tilde{V}$ (km/s)', r'$\tilde{W}$ (km/s)']
    fig, axs = plt.subplots(5, 5, figsize=(11.,11.))
    for i in range(5):
        axs[4][i].set_xlabel(labels_vec[i])
        axs[i][0].set_ylabel(labels_vec[i+1])
        for j in range(5):
            axs[i][j].tick_params(direction='in')
            axs[i][j].xaxis.set_ticks_position('both')
            axs[i][j].yaxis.set_ticks_position('both')
            axs[i][j].xaxis.set_major_locator(plt.MaxNLocator(2))
            axs[i][j].yaxis.set_major_locator(plt.MaxNLocator(2))
            if i>0:# or i==j:
                axs[j][i].set_yticklabels([])
            if j<4:
                axs[j][i].set_xticklabels([])
        for j in range(5):
            if j>i:
                axs[i][j].set_visible(False)
    for i in range(5):
        #axs[i][i].hist(xyzs[:,i], 10, histtype='step', color='k')
        #axs[i][i].set_yticks([])
        for j in range(i,5):
            axs[j][i].plot(xyzs[:,i], xyzs[:,j+1], 'k.', alpha=0.3)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    if save_path!='':
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
    plt.show()
    return 


# Some examples below:

# make_posterior_plot(save_path='../Figs/posterior.png')
# make_corr_plot(res_path='../Results/HMCchain_20-1000_inclined_300_2_1kpc_90deg_updated.npz', save_path='')
# make_corr_plot(res_path='../Results/HMCchain_20-1000_slow_300_0_1kpc_90deg_updated.npz', save_path='../Figs/correlations_S1.png')
# make_corr_plot(res_path='../Results/HMCchain_20-1000_corotB_300_2_1kpc_180deg_updated.npz', save_path='')
# make_corr_plot(res_path='../Results/HMCchain_20-1000_slow_300_0_1kpc_90deg_updated.npz', save_path='../Figs/correlations_S1.png')
# make_corr_plot(res_path='../Results/HMCchain_20-1000_corotB_300_0_1kpc_180deg_updated.npz', save_path='../Figs/correlations_S2.png')
# make_corr_plot(res_path='../Results/HMCchain_20-1000_corotC_300_0_1kpc_180deg_updated.npz', save_path='../Figs/correlations_S3.png')
# make_corr_plot(res_path='../Results/HMCchain_20-1000_inclined_300_0_1kpc_90deg.npz', save_path='../Figs/correlations_S4.png')
# make_err_plot()
  
# make_data_plot(res_path='../Results/HMCchain_20-1000_slow_300_0_1kpc_90deg_updated.npz', save_path='../Figs/data_and_orbit_S1.png')
# make_data_plot(res_path='../Results/HMCchain_20-1000_corotB_300_2_1kpc_180deg_updated.npz', save_path='')
# make_data_plot(res_path='../Results/HMCchain_20-1000_corotC_300_2_1kpc_180deg_updated.npz', save_path='')
# make_data_plot(res_path='../Results/HMCchain_20-1000_inclined_300_2_1kpc_90deg_updated.npz', save_path='')
  
# make_stream_coord_plot('../Results/HMCchain_20-1000_slow_300_0_1kpc_90deg_updated.npz', save_path='../Figs/coords_S1.png')
# make_stream_coord_plot('../Results/HMCchain_20-1000_corotB_300_0_1kpc_180deg_updated.npz', save_path='../Figs/coords_S2.png')
# make_stream_coord_plot('../Results/HMCchain_20-1000_corotC_300_0_1kpc_180deg_updated.npz', save_path='../Figs/coords_S3.png')
# make_stream_coord_plot('../Results/HMCchain_20-1000_inclined_300_0_1kpc_90deg.npz', save_path='../Figs/coords_S4.png')
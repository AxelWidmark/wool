import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
pi=math.pi
from scipy.interpolate import interp1d
from scipy.optimize import minimize



mpl.rcParams['font.serif'] = "Times"
mpl.rcParams["mathtext.fontset"] = "dejavuserif"
mpl.rcParams['font.family'] = "serif"
mpl.rcParams.update({'font.size': 12})
effective_dens_derivRterms = -0.00222176


def vz_of_z(z,params):
    vz_0,rho_a,r_a,rho_b = params
    phi = 2e6/37.*(  rho_a*np.log(math.cosh(z/r_a))*r_a**2. + rho_b*z**2./2.  )
    vz = -np.sqrt(vz_0**2.-2.*phi)
    return vz
def rho_of_z(z,params):
    vz_0,rho_a,r_a,rho_b = params
    rho = rho_a*math.cosh(z/r_a)**(-2.) + rho_b
    return rho


# run MCMC on the stream
def inference_on_stream():
    # this is a Metropolis-Hastings MCMC sampler
    class sampler():
      
        def __init__(self,lnprob,dim,p0,initialstepcovar):
            self.chain = [p0]
            if np.shape(initialstepcovar)==(dim,dim):
                self.stepcovar = initialstepcovar
            elif np.shape(initialstepcovar)==(dim,):
                self.stepcovar=np.zeros((dim,dim))
                for i in range(dim):
                    self.stepcovar[i][i]=initialstepcovar[i]
            else:
                raise ValueError('Stepsize vector/matrix has incorrect dimension')
            self.lnprob = lnprob
            self.dim = dim
    
    
        def run(self,n,printsteps=False):
            # tail is the end of the new chain
            tail = np.zeros((n,self.dim))
            params0 = self.chain[-1]
            lh0 = self.lnprob(params0)
            accepts=0
            params1 = np.zeros(self.dim)
            # ITERATE
            for i in range(n):
                params1 = np.random.multivariate_normal(params0,self.stepcovar)
                lh1 = self.lnprob(params1)
                lhratio = np.exp(lh1-lh0)
                if lhratio>1. or lhratio>np.random.rand():
                    accepts += 1
                    tail[i] = params1
                    params0 = list(params1)
                    lh0 = lh1
                else:
                    tail[i] = params0
                if printsteps:
                    print(params0)
            self.chain = np.concatenate((self.chain,tail))
            return tail,float(accepts)/float(n)
    
    
        def run_endpoint_only(self,n):
            params0 = self.chain[-1]
            lh0 = self.lnprob(params0)
            accepts=0
            params1 = np.zeros(self.dim)
            # ITERATE
            for i in range(n):
                params1 = np.random.multivariate_normal(params0,self.stepcovar)
                lh1 = self.lnprob(params1)
                lhratio = np.exp(lh1-lh0)
                if lhratio>1. or lhratio>np.random.rand():
                    accepts += 1
                    params0 = list(params1)
                    lh0 = lh1
            return params0,float(accepts)/float(n)
    
    
        def burnin(self,n):
            self.clear_chain()
            tail,acc = self.run(n)
            if acc<.1 or acc>.9:
                print("BURNIN WARNING, acc =",acc)
                print(self.stepcovar)
            self.set_stepcovar()
            self.clear_chain()
            return acc
    
    
        def set_stepcovar(self):
            self.stepcovar=np.cov(self.chain,rowvar=False)
    
    
        def clear_chain(self):
            self.chain=[self.chain[-1]]
            return self.chain
    
    
        def get_chain(self):
            return self.chain
    
    
        def get_stepcovar(self):
            return self.stepcovar
    # load the z and vz coordinates of a perfect stream
    from LoadStream import load_stream
    ls = load_stream('perfect_stream')
    coords = ls.get_stream_coords()
    # generate 50 random indices
    rand_ints = np.random.randint(0,len(coords),100)
    orb_x, orb_y, orb_z, orb_vx, orb_vy, orb_vz = np.transpose(coords[rand_ints])
    plt.plot(orb_z, orb_vz,'.')
    plt.show()
    from gala.potential import MilkyWayPotential
    pot=MilkyWayPotential()
    rho_dots = [pot.density([sss[0],sss[1],sss[2]])/1e9 for sss in np.transpose([orb_x,orb_y,orb_z])]
    # function to minimize
    def func2minimize(params):
        vz_0,rho_a,r_a,rho_b = params
        if rho_a<0. or rho_b<0. or r_a<0.1 or r_a>.4:
            return np.inf
        res = 0.
        for i in range(len(orb_z)):
            vdiff = orb_vz[i]-vz_of_z(orb_z[i],params)
            vsigma = .1
            res += vdiff**2./(2.*vsigma**2.)
        return res
    # initial guess
    p0 = [-202.,  0.03,  0.18,  0.06]
    # minimize
    res = minimize(func2minimize, p0, method='Nelder-Mead', tol=1e-8)
    print('Minimization results:',res.x)
    ln_prob = lambda params: -func2minimize(params)
    p0 = res.x
    stepcovar0 = [1e-6, 1e-8, 1e-8, 1e-8]
    s = sampler(ln_prob, 4, p0, stepcovar0)
    s.run(500)
    s.burnin(1000)
    s.burnin(2000)
    s.burnin(3000)
    tail,acc = s.run(100000)
    print('MCMC acceptance rate:', acc)
    # run the same thing but without varying shape
    ln_prob = lambda params: -func2minimize([params[0], params[1]*p0[1]/(p0[1]+p0[3]), p0[2], params[1]*p0[3]/(p0[1]+p0[3])])
    stepcovar0 = [1e-6, 1e-8]
    s = sampler(ln_prob, 2, [p0[0],p0[1]+p0[3]], stepcovar0)
    s.run(500)
    s.burnin(1000)
    s.burnin(2000)
    s.burnin(3000)
    tail2,acc = s.run(100000)
    print('MCMC acceptance rate:', acc)
    np.savez('../Data/perfect_orbit_results', orb_x=orb_x, orb_y=orb_y, orb_z=orb_z, orb_vx=orb_vx, orb_vy=orb_vy, orb_vz=orb_vz, tail=tail[::10], tail2=tail2[::10], rho_dots=rho_dots, param_min=p0)
    return 

# make Fig. 1 of the paper
def make_fit_plot():
    npzfile = np.load('../Data/perfect_orbit_results.npz')
    orb_z = npzfile['orb_z']
    orb_vz = npzfile['orb_vz']
    rho_dots = npzfile['rho_dots']
    tail = npzfile['tail']
    param_min = npzfile['param_min']
    
    reduced_tail = tail[np.random.randint(0,len(tail),20)]
    
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.8,6))
    plt.subplots_adjust(hspace=0.)
    
    z_vec = np.linspace(-0.4, 0.4, 1000)
    vz_vec = [vz_of_z(z, param_min) for z in z_vec]
    
    for params_t in reduced_tail[0:5]:
        vz_vec = [vz_of_z(z, params_t) for z in z_vec]
        ax1.plot(z_vec, vz_vec, 'k-', alpha=0.1, linewidth=2)
    ax1.plot(z_vec, [vz_of_z(z, param_min) for z in z_vec], linewidth=2)
    ax1.plot(orb_z, orb_vz, 'r.')
    
    for params_t in reduced_tail:
        rho_vec = [rho_of_z(z, params_t)+effective_dens_derivRterms for z in z_vec]
        ax2.plot(z_vec, rho_vec, 'k-', alpha=0.1, linewidth=2)
    ax2.plot(z_vec, [rho_of_z(z, param_min)+effective_dens_derivRterms for z in z_vec], linewidth=2)
    ax2.plot(orb_z, rho_dots, 'r.')
    
    ax1.set_yticks([-201.,-201.5,-202])
    ax2.set_yticks([0.02,0.06,0.10])
    #ax1.grid(True)
    for ax in (ax1,ax2):
        ax.set_xticks([-0.4,-0.2,0.,0.2,0.4])
        #ax.set_xticklabels(['-0.4','-0.2','0','0.2','0.4'])
        ax.set_xlim([-0.4,0.4])
        ax.tick_params(direction='in')
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
    ax1.set_xticklabels([])
    
    ax1.set_ylabel(r'$\tilde{W}$ (km/s)')
    ax2.set_xlabel(r'$\tilde{Z}$ (kpc)')
    ax2.set_ylabel(r'$\rho$ (M$_{\odot}/$pc$^{3}$)')
    
    ax1.plot([], [], 'r.', label='True values of stars')
    ax1.plot([], [], '#1f77b4', linewidth=2, label='Best fit')
    ax1.plot([], [], 'k-', alpha=0.1, linewidth=2, label='Posterior realization')
    ax1.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.)
    
    plt.savefig('../Figs/perf_orb_fit.png', bbox_inches='tight', dpi=600)
    plt.show()


# make Fig. 2 of the paper
def make_surfdens_plot():
    npzfile = np.load('../Data/perfect_orbit_results.npz')
    orb_z = npzfile['orb_z']
    orb_vz = npzfile['orb_vz']
    rho_dots = npzfile['rho_dots']
    tail = npzfile['tail']
    tail2 = npzfile['tail2']
    param_min = npzfile['param_min']
    intMs = []
    z_vec = np.linspace(0.,0.24,2401)
    f_rho = interp1d(np.abs(orb_z), rho_dots[:,0], fill_value="extrapolate")
    true_value = 0.2*np.sum([f_rho(z) for z in z_vec])
    for param_t in tail:
        intMs.append(0.2*np.sum([rho_of_z(z, param_t)+effective_dens_derivRterms for z in z_vec]))
    print(np.mean(intMs),np.median(intMs),np.cov(intMs)**.5)
    f, ax = plt.subplots(1, 1, figsize=(5,4))
    print(true_value)
    ax.plot([true_value,true_value], [0.,2300.], 'k--')
    ax.hist(intMs, np.linspace(30.,50.,41), histtype='step')
    ax.tick_params(direction='in')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.set_xlim([33.5,40.5])
    ax.set_ylim([0.,2300])
    ax.set_yticks([])
    ax.set_xlabel(r'$\Sigma_{240}$ (M$_{\odot}/$pc$^{2}$)')
    ax.set_ylabel('Marginalized posterior density')
    plt.savefig('../Figs/perf_orb_surfdens.png', bbox_inches='tight', dpi=600)
    plt.show()
    return 
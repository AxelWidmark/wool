from LoadStream import load_stream
from TfFunctions import tffunc
import matplotlib.pyplot as plt
import numpy as np
import corner
pi = np.pi



# this function starts an MCMC run, with a thorough burn-in phase
def run_chain_with_burnin(which_stream, number_of_stars, stream_distance, ang_0, nametag='', stellar_start=0, zlim=0.5):
    path2stream = '../GeneratedStreams/mock_stream_' +which_stream+'.npz'
    print(path2stream)
    s = load_stream(path2stream, plot=False, zlim=zlim)
    
    xyzs = s.get_stream_coords()
    s.make_data(stream_distance=stream_distance, ang_0=ang_0, number_of_stars=number_of_stars, stellar_start=stellar_start, plot=False)
    p0 = s.suggest_p0()
    obs_list = s.get_data()
    fx = s.get_fx()
    z_sun = s.get_z_sun()
    sun_vel = s.get_sun_vel()
    print('p0:',p0)
    
    # load tensorflow functions class
    f = tffunc(obs_list, fx, z_sun, sun_vel)
    pN = np.concatenate((p0, 5.*np.ones(len(obs_list))))
    
    p1, nuisance_params = f.minimize_posterior(pN, number_of_iterations=10000, print_gap=1000, plot_gap=None, numTFTthreads=1, learning_rate=1e-3)
    pN = np.concatenate((p1, nuisance_params))
    print(p1)
    ss_0 = np.array([1e-4,1e-6,1e-3,1e-4,1e-5,1e-5,1e-5,1e-2,1e-2,1e-2,1e-2,1e-2,1e-3,1e-4,1e-5,1e-5])
    ssN = np.concatenate((ss_0,1e-3*np.ones(len(obs_list))))
    
    print('burnin')
    pN,ssN = f.burnin_with_npp(pN, ssN, steps=1e5, burnin_steps=1e4, num_adaptation_steps=4e3, num_leapfrog_steps=1, iterations=10)
    print('start final HMC')
    param_samples, samples, log_prob, step_size, is_accepted = f.run_HMC(pN, steps=1e5, burnin_steps=1e4, num_adaptation_steps=4e3, num_leapfrog_steps=10, step_size_start=ssN)
    
    step_size = ssN
    print(is_accepted)
    print('\n\n\n')
    np.savez('../Results/HMCchain_'+which_stream+'_'+str(number_of_stars)+'_'+str(stellar_start)+nametag, log_prob=log_prob, obs_list=obs_list, param_samples=param_samples, samples=samples, step_size=step_size, is_accepted=is_accepted, xyzs=xyzs)
    return 


# this function extends an already existing MCMC chain, in case the burn-in was poor or you just want to run it for longer
def update_old_chain(path2chain):
    import pandas as pd
    from gala.potential import MilkyWayPotential
    pot=MilkyWayPotential()
    import astropy.units as u
    import gala.potential as gp
    import gala.dynamics as gd
    
    npzfile = np.load(path2chain+'.npz')
    obs_list = npzfile['obs_list']
    samples = npzfile['samples']
    param_samples = npzfile['param_samples']
    xyzs = npzfile['xyzs']
    
    df_stream = pd.DataFrame(xyzs, columns = ['x', 'y', 'z', 'vx', 'vy', 'vz'])
    r_median = np.median(np.sqrt(df_stream['x']**2.+df_stream['y']**2.+df_stream['z']**2.))
    fx = float( (pot.value([r_median+0.01,0.,0.])-pot.value([r_median-0.01,0.,0.]))[0]/0.02/(u.kpc/u.Myr)**2.*3.103e-11 )
    z_sun = 0.015
    sun_vel = np.array([-11.1,245.8,7.2])
    
    f = tffunc(obs_list, fx, z_sun, sun_vel)
    
    pN = np.concatenate( (param_samples[-1],samples[-1,16:16+len(obs_list)]) )
    ssN = 1e-1 * np.sqrt( [np.cov(ss) for ss in np.transpose(samples)] )
    
    print('start HMC update')
    param_samples, samples, log_prob, step_size, is_accepted = f.run_HMC(pN, steps=5e5, burnin_steps=1e4, num_adaptation_steps=9e3, num_leapfrog_steps=10, step_size_start=ssN)
    
    step_size = ssN
    print(is_accepted)
    print('\n\n\n')
    np.savez(path2chain+'_updated', log_prob=log_prob, obs_list=obs_list, param_samples=param_samples, samples=samples, step_size=step_size, is_accepted=is_accepted, xyzs=xyzs)
    return


# Example:
# run_chain_with_burnin(disps+'_inclined', number_of_stars, 1., pi/2., nametag='_1kpc_90deg', zlim=0.5, stellar_start=stellar_start)
        
import math
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
pi=math.pi
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import seaborn as sns
import astropy.units as u
import gala.potential as gp
import gala.dynamics as gd
from gala.potential import MilkyWayPotential
pot=MilkyWayPotential()



# v_RV limit could be ~14 (really 14.5 but stellar crowding could make it worse)
Gmags = np.arange(12,21)
# parallax, proper motion
uncertainties = 1e-3*np.array([ [6.7, 10.3, 16.5, 26.6, 43.6, 74.5, 137, 280, 627], [3.5, 5.4, 8.7, 14.0, 22.9, 39.2, 72.3, 147, 330] ])
# functions of uncertainty, interpolated over apparent G-mags
log10_par_err_of_G = interp1d(Gmags, np.log10(uncertainties[0]), bounds_error=False, fill_value=(np.log10(uncertainties[0][0]),np.log10(uncertainties[0][-1])))
log10_mu_err_of_G = interp1d(Gmags, np.log10(uncertainties[1]), bounds_error=False, fill_value=(np.log10(uncertainties[1][0]),np.log10(uncertainties[1][-1])))


# this class loads a synthesised stream in stream phase-space coords, and makes it into data
class load_stream():
    
    
    # initialisation, most importantly give the stream path
    def __init__(self, path2stream, zlim=0.4, plot=False):
        self.zlim = zlim
        if path2stream=='perfect_stream':
            from MakeMockStream import make_perfect_stream
            xyzs = make_perfect_stream()
        else:
            npzfile = np.load(path2stream)
            xyzs = npzfile['xyzs']
        self.df_stream = pd.DataFrame(xyzs, columns = ['x', 'y', 'z', 'vx', 'vy', 'vz'])
        self.df_stream = self.df_stream[ (np.abs(self.df_stream['z'])<zlim) ]
        self.df_stream = self.df_stream[ (np.abs(np.median(self.df_stream['x'])-self.df_stream['x'])<1.) ]
        r_median = np.median(np.sqrt(self.df_stream['x']**2.+self.df_stream['y']**2.+self.df_stream['z']**2.))
        self.fx = float( (pot.value([r_median+0.01,0.,0.])-pot.value([r_median-0.01,0.,0.]))[0]/0.02/(u.kpc/u.Myr)**2.*3.103e-11 )
        self.rot_vel = np.sqrt(3.086e16*r_median*self.fx)
        print('fx:', self.fx, '(km/s^2)')
        print('Rot.vel.:', self.rot_vel, '(km/s)')
        print('Total size:', self.df_stream.shape[0])
        if plot:
            self.plot_stream()
        return
            
    
    # make data with uncertainteis and errors
    def make_data(self, stream_distance=2., ang_0=pi/2., z_sun=0.015, sun_vel=np.array([-11.1,245.8,7.2]), number_of_stars=None, stellar_start=0, plot=False):
        if number_of_stars==None:
            number_of_stars = self.df_stream.shape[0]
            stellar_thinning = 1
        else:
            stellar_thinning = max( int(self.df_stream.shape[0]/number_of_stars), 1)
        if os.path.exists('../GeneratedStreams/mock_isochrone_Gmags.npy'):
            isochrones_appGs = np.load('../GeneratedStreams/mock_isochrone_Gmags.npy')
        else:
            from isochrones import get_ichrone
            from isochrones.priors import ChabrierPrior
            tracks = get_ichrone('mist', tracks=True)
            N = 10000
            masses = ChabrierPrior().sample(N)
            feh = -2.  # Fe/H
            age = np.log10(12e9)  # 12 Gyr
            distance = 10. # 10 pc
            df = tracks.generate(masses, age, feh, distance=distance)
            df = df.dropna()
            df = df[df['G_mag']<20.-8.]
            isochrones_appGs = df['G_mag'].values
            np.save('../GeneratedStreams/mock_isochrone_Gmags', isochrones_appGs)
        self.ang_0 = ang_0
        self.z_sun = z_sun
        sun_pos = [np.median(self.df_stream['x'])-np.cos(ang_0)*stream_distance, np.median(self.df_stream['y'])-np.sin(ang_0)*stream_distance, self.z_sun]
        self.sun_vel = sun_vel
        self.stream_distance = stream_distance
        obs_list = []
        for index in self.df_stream.index[stellar_start:stellar_start+stellar_thinning*number_of_stars:stellar_thinning]:
            rel_pos = np.array( [self.df_stream['x'][index]-sun_pos[0], self.df_stream['y'][index]-sun_pos[1], self.df_stream['z'][index]-sun_pos[2]] )
            dist = np.linalg.norm( rel_pos )
            appG = 30.
            G_diff = 5.*(np.log10(1e3*dist)-1.)
            while appG>20.:
                appG = np.random.choice(isochrones_appGs) + G_diff
            par_err = 10.**log10_par_err_of_G(appG)
            mu_err = 10.**log10_mu_err_of_G(appG)
            vlos_err = 0.3
            par = 1./dist + par_err*np.random.normal()
            if rel_pos[1]>0.:
                l = np.arccos(rel_pos[0]/np.sqrt(rel_pos[0]**2.+rel_pos[1]**2.))
            else:
                l = 2.*pi -np.arccos(rel_pos[0]/np.sqrt(rel_pos[0]**2.+rel_pos[1]**2.))
            b = np.arcsin(rel_pos[2]/dist)
            rel_vel = [self.df_stream['vx'][index]-sun_vel[0], self.df_stream['vy'][index]-sun_vel[1], self.df_stream['vz'][index]-sun_vel[2]]
            mul  = ( -np.sin(l)*rel_vel[0] +np.cos(l)*rel_vel[1] ) / (4.74057*dist)                         + mu_err*np.random.normal()
            mub  = ( -np.sin(b)*np.cos(l)*rel_vel[0] -np.sin(b)*np.sin(l)*rel_vel[1] +np.cos(b)*rel_vel[2] ) / (4.74057*dist)   + mu_err*np.random.normal()
            vlos = ( +np.cos(b)*np.cos(l)*rel_vel[0] +np.cos(b)*np.sin(l)*rel_vel[1] +np.sin(b)*rel_vel[2] )                    + vlos_err*np.random.normal()
            obs_list.append([par,l,b,mul,mub,vlos,par_err,mu_err,vlos_err,appG])
        obs_list = np.array(obs_list)
        self.df_data = pd.DataFrame(obs_list, columns=['par','l','b','mul','mub','vlos','par_err','mu_err','vlos_err','appG'])
        print("Size of obs. list:", np.shape(obs_list))
        if plot:
            self.plot_data()
        return
    
    
    # get data and stream phase-space coordinates
    def get_data(self, coords=['par','l','b','mul','mub','vlos','par_err','mu_err','vlos_err','appG']):
        return np.array( self.df_data[coords].values )
    def get_stream_coords(self, coords=['x', 'y', 'z', 'vx', 'vy', 'vz']):
        return np.array( self.df_stream[coords].values )
    
    
    # get gravitational force acting towards the Galactic centre
    def get_fx(self):
        return self.fx
    
    # get some solar properties
    def get_z_sun(self):
        return self.z_sun
    def get_sun_vel(self):
        return self.sun_vel
    
    # get rotational matrices (denoted M, in the paper appendix)
    def get_rotMs(self):
        obs_list = self.get_data()
        rot_matrices = np.array( [ [                                    \
            [-np.sin(l), +np.cos(l), 0.],           \
            [-np.sin(b)*np.cos(l), -np.sin(b)*np.sin(l), +np.cos(b)],   \
            [+np.cos(b)*np.cos(l), +np.cos(b)*np.sin(l), +np.sin(b)]    \
            ] for l,b in obs_list[:,1:3]] )
        rot_matrices_inv = np.array( [np.linalg.inv(m) for m in rot_matrices] )
        return rot_matrices, rot_matrices_inv
        
    # suggest a VERY ROUGH starting point for the MCMC
    def suggest_p0(self):
        l_0 = np.polyfit(self.df_data['b'], self.df_data['l'], 2)[-1]
        return [0.1, self.stream_distance, self.fx, np.median(self.df_stream['vx']), np.median(self.df_stream['vy']), np.median(self.df_stream['vz']), l_0, 8., 8., 8., 0.3, 0.3, self.z_sun, self.sun_vel[0], self.sun_vel[1], self.sun_vel[2]]
    
    # plot the data
    def plot_data(self, plot_height=2.):
        sns.pairplot(self.df_data, height=plot_height, markers=".")
        plt.show()
        return
    
    # plot the stream phase-space coordinates
    def plot_stream(self, plot_height=2., number_of_stars=None, stellar_start=0):
        if number_of_stars==None:
            stellar_thinning = 1
            sns.pairplot(self.df_stream, height=plot_height, markers=".")
            plt.show()
        else:
            stellar_thinning = max( int(self.df_stream.shape[0]/number_of_stars), 1)
            sns.pairplot(self.df_stream[stellar_start:-1:stellar_thinning], height=plot_height, markers=".")
            plt.show()
        return
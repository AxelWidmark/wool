import math
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
pi=math.pi

import pandas as pd
import seaborn as sns

import tensorflow as tf
import tensorflow_probability as tfp
numTFThreads = 2
tf.compat.v1.disable_eager_execution()

# these are the variable bounds for the population parameters in TensorFlow code, for example preventing TF from searching negative dispersions
# these bounds are NOT THE PRIOR PROBABILITY
var_bounds = np.array([
    [0., .3],           # Rho
    [0., 5.],           # Dist_0
    [0., 5e-13],        # Fx
    [-500., 500.],      # Vx_0
    [-500., 500.],      # Vy_0
    [-500., 500.],      # Vz_0
    [-pi/2., 5.*pi/2.], # Ang_0
    [0., 10.],        # Sigma_vx
    [0., 10.],        # Sigma_vy
    [0., 10.],        # Sigma_vz
    [0., .5],         # Sigma_x
    [0., .5],         # Sigma_y
    [-0.05, 0.05],      # Z_sun
    [-300., 300.],      # Vx_sun
    [-300., 300.],      # Vy_sun
    [-50., 50.]         # Vz_sun
])

# the variable space of the posterior is expressed via sigmoid functions and the bounds above
# these functions correspond to variable transformations
def sigmoid(x):
    return 1./(1.+np.exp(-x))
def sigmoid_inv(x):
    return -np.log(1./x-1.)
def p0_2_vector(p0):
    return sigmoid_inv((p0-var_bounds[:,0])/(var_bounds[:,1]-var_bounds[:,0]))
def v_2_params(v):
    if np.shape(v)==np.shape(var_bounds[:,0]):
        return var_bounds[:,0] + (var_bounds[:,1]-var_bounds[:,0])*sigmoid(v)
    else:
        return np.array([ var_bounds[:,0] + (var_bounds[:,1]-var_bounds[:,0])*sigmoid(vv) for vv in v ])


# this is the main class of this whole project, the TensorFlow functions in which the model is written
class tffunc():
    
    # initialise by giving:
    # obs_list: the list of data for the stream
    # fx: the mode of prior for the graviational force acting on the stream in the direction of the Gal. centre
    # z_sun: the mode of the prior for the sun's height w.r.t. the Gal. plane
    # sun_vel: the mode of the prior for the sun's velocities in the Gal. rest frame
    # vlos_appG_limit (optional): the app.mag. limit for radial velocities
    def __init__(self, obs_list, fx, z_sun, sun_vel, vlos_appG_limit=14.):
        # this sets a bunch of constants (should be fairly self-explanatory given their variable names)
        self.obs_list = obs_list
        self.obs_list_len = len(self.obs_list)
        self.Fx_true = tf.constant(fx, dtype=tf.float32)
        self.Par_obs_vec = tf.constant(obs_list[:,0], dtype=tf.float32)
        self.L_vec = tf.constant(obs_list[:,1], dtype=tf.float32)
        self.B_vec = tf.constant(obs_list[:,2], dtype=tf.float32)
        self.Mul_vec = tf.constant(obs_list[:,3], dtype=tf.float32)
        self.Mub_vec = tf.constant(obs_list[:,4], dtype=tf.float32)
        self.Vlos_vec = tf.constant(obs_list[:,5], dtype=tf.float32)
        self.Par_err_vec = tf.constant(obs_list[:,6], dtype=tf.float32)
        self.Mul_err_vec = tf.constant(obs_list[:,7], dtype=tf.float32)
        self.Mub_err_vec = tf.constant(obs_list[:,7], dtype=tf.float32)
        self.Vlos_err_vec = tf.constant(obs_list[:,8], dtype=tf.float32)
        self.Vx_sun_true = tf.constant(sun_vel[0], dtype=tf.float32)
        self.Vy_sun_true = tf.constant(sun_vel[1], dtype=tf.float32)
        self.Vz_sun_true = tf.constant(sun_vel[2], dtype=tf.float32)
        self.Z_sun_true = tf.constant(z_sun, dtype=tf.float32)
        rot_matrices = np.array( [ [                                    \
            [-np.sin(l), +np.cos(l), 0.],           \
            [-np.sin(b)*np.cos(l), -np.sin(b)*np.sin(l), +np.cos(b)],   \
            [+np.cos(b)*np.cos(l), +np.cos(b)*np.sin(l), +np.sin(b)]    \
            ] for l,b in self.obs_list[:,1:3]] )
        rot_matrices_inv = np.array( [np.linalg.inv(m) for m in rot_matrices] )
        self.Rot_matrices = tf.constant(rot_matrices, dtype=tf.float32)
        self.Rot_matrices_inv  = tf.constant(rot_matrices_inv, dtype=tf.float32)
        self.V_sun_true = tf.stack([self.Vx_sun_true, self.Vy_sun_true, self.Vz_sun_true])
        self.Mlmbvr_obs_vec = tf.transpose(tf.stack([self.Mul_vec, self.Mub_vec, self.Vlos_vec]))
        self.Sigma_mlmbvr_obs_vec = tf.transpose( tf.stack(      \
            [ tf.stack( [self.Mul_err_vec**2, tf.zeros(tf.shape(self.Mul_err_vec), dtype=tf.float32), tf.zeros(tf.shape(self.Mul_err_vec), dtype=tf.float32)] ),  \
              tf.stack( [tf.zeros(tf.shape(self.Mul_err_vec), dtype=tf.float32), self.Mub_err_vec**2, tf.zeros(tf.shape(self.Mul_err_vec), dtype=tf.float32)] ),  \
              tf.stack( [tf.zeros(tf.shape(self.Mul_err_vec), dtype=tf.float32), tf.zeros(tf.shape(self.Mul_err_vec), dtype=tf.float32), self.Vlos_err_vec**2] )   \
            ] ), [2,0,1] )
        print('Stars with v_RV:', np.where(obs_list[:,9]<vlos_appG_limit)[0])
        self.Vlos_incl_indices = tf.constant(np.where(obs_list[:,9]<vlos_appG_limit)[0], dtype=tf.int32)
        self.Vlos_excl_indices = tf.constant(np.where(obs_list[:,9]>=vlos_appG_limit)[0], dtype=tf.int32)
        self.Var_bounds = tf.constant(var_bounds, dtype=tf.float32)
    
    # transforms the vector of parameters to the 16 model parameters, via the sigmoid function
    @tf.function
    def Vector_2_Params(self, Vector):
        return self.Var_bounds[:,0] + (self.Var_bounds[:,1]-self.Var_bounds[:,0])*tf.sigmoid(Vector)
    
    # gravitational potential as function of height above the plane
    # (this is expressed as the mid-plane density at Z=0, rather than Sigma_240 as in the paper -- they differ by a constant)
    @tf.function
    def phi_of_z(self, rho, z):
        return 2e6/37.*(  rho*0.8*tf.math.log(tf.cosh(z/2.3e-01))*2.3e-01**2. + rho*0.2*z**2./2.  )
    
    # this function takes the population parameters and returns a vector of the orbit trajectory phase-space coordinates
    @tf.function
    def ideal_trajectory(self, Params, zlims=[-0.4,0.4]):
        Rho, Dist_0, Fx, Vx_0, Vy_0, Vz_0, Ang_0, Sigma_vx, Sigma_vy, Sigma_vz, Sigma_x, Sigma_y, Z_sun, Vx_sun, Vy_sun, Vz_sun   \
            = tf.unstack(tf.cast(Params, tf.float32), 16)
        X_0 = tf.cos(Ang_0)*Dist_0
        Y_0 = tf.sin(Ang_0)*Dist_0
        V_sun = tf.stack([Vx_sun, Vy_sun, Vz_sun])
        Z_ideal = tf.constant(np.linspace(zlims[0],zlims[1],101), dtype=tf.float32)
        Z_sum = Z_ideal+Z_sun[None]
        Phi_Z_ideal = self.phi_of_z(Rho, Z_sum)
        Vz_ideal_prelim_vec =  tf.stack( [tf.sign(Vz_0[None]) * tf.sqrt(Vz_0[None]**2 -2.*self.phi_of_z(Rho, z_frac*(Z_ideal+Z_sun[None]))) for z_frac in np.linspace(0., 1., 5)] )
        Time_ideal = tf.math.reduce_mean( (Z_ideal[None,:])/Vz_ideal_prelim_vec, [0])
        X_ideal = X_0[None] + Vx_0[None]*Time_ideal + 3.086e16*Fx*Time_ideal**2/2.
        Y_ideal = Y_0[None] + Vy_0[None]*Time_ideal
        Vx_ideal = Vx_0[None] + 3.086e16*Fx*Time_ideal
        Vy_ideal = Vy_0*tf.ones(tf.shape(Vx_ideal), dtype=tf.float32)
        Vz_ideal = tf.sign(Vz_0[None]) * tf.sqrt(Vz_0[None]**2 -2.*Phi_Z_ideal)
        Dist_ideal = tf.sqrt(X_ideal**2+Y_ideal**2+Z_ideal**2)
        Vel_ideal = tf.transpose( tf.stack([Vx_ideal, Vy_ideal, Vz_ideal]) )
        L_ideal = pi + tf.sign(Y_ideal)*( tf.math.acos(X_ideal/tf.sqrt(X_ideal**2+Y_ideal**2)) -pi )
        B_ideal = tf.math.asin(Z_ideal/Dist_ideal)
        Rot_ideal = tf.transpose( tf.stack([tf.stack( [-tf.sin(L_ideal),  tf.cos(L_ideal), tf.zeros(tf.shape(L_ideal), dtype=tf.float32)] ), \
                                            tf.stack( [-tf.sin(B_ideal)*tf.cos(L_ideal), -tf.sin(B_ideal)*tf.sin(L_ideal), tf.cos(B_ideal)] ), \
                                            tf.stack( [ tf.cos(B_ideal)*tf.cos(L_ideal),  tf.cos(B_ideal)*tf.sin(L_ideal), tf.sin(B_ideal)] ) ] ), [2,0,1])
        Par_ideal = 1./Dist_ideal
        Vlvbvr_ideal = tf.linalg.matvec(Rot_ideal, Vel_ideal-V_sun[None,:])
        Mul_ideal = Vlvbvr_ideal[:,0]/(4.74057*Dist_ideal)
        Mub_ideal = Vlvbvr_ideal[:,1]/(4.74057*Dist_ideal)
        Vlos_ideal = Vlvbvr_ideal[:,2]
        Obs_ideal = tf.transpose( tf.stack([Par_ideal, L_ideal, B_ideal, Mul_ideal, Mub_ideal, Vlos_ideal]) )
        return Obs_ideal
    
    # the posterior density of model parameters and distance nuisance parameters
    @tf.function
    def unnormalized_posterior_log_prob_with_nuisance_par_params(self, Params, Nuisance_params):
        # rho, dist_0, fx, vx_0, vy_0, vz_0, ang_0, sigma_vx, sigma_vy, sigma_vz, sigma_x, sigma_y, z_sun, vx_sun, vy_sun, vz_sun
        Rho, Dist_0, Fx, Vx_0, Vy_0, Vz_0, Ang_0, Sigma_vx, Sigma_vy, Sigma_vz, Sigma_x, Sigma_y, Z_sun, Vx_sun, Vy_sun, Vz_sun   \
            = tf.unstack(Params, 16)
        Par_vec = self.Par_obs_vec + Nuisance_params*self.Par_err_vec
        Dist_vec = 1./Par_vec
        Log_nuisance_jacobian = tf.math.log( self.Par_err_vec / ( 2.*Par_vec ) )
        X_0 = tf.cos(Ang_0)*Dist_0
        Y_0 = tf.sin(Ang_0)*Dist_0
        V_sun = tf.stack([Vx_sun, Vy_sun, Vz_sun])
        X_vec = tf.cos(self.L_vec) * tf.cos(self.B_vec) * Dist_vec
        Y_vec = tf.sin(self.L_vec) * tf.cos(self.B_vec) * Dist_vec
        Z_vec = tf.sin(self.B_vec) * Dist_vec
        # 3.086e16 is km/kpc
        Phi_Z_vec = self.phi_of_z(Rho, Z_vec+Z_sun[None])
        Vz_prelim_vec_vec =  tf.stack( [tf.sign(Vz_0[None]) * tf.sqrt(Vz_0[None]**2 -2.*self.phi_of_z(Rho, z_frac*(Z_vec+Z_sun[None]))) for z_frac in np.linspace(0., 1., 5)] )
        Time_vec = tf.math.reduce_mean( (Z_vec[None,:]+Z_sun[None,None])/Vz_prelim_vec_vec, [0])
        Vx_mean_vec = Vx_0[None] + 3.086e16*Fx*Time_vec
        X_mean_vec = X_0[None] + Vx_0[None]*Time_vec + 3.086e16*Fx[None]*Time_vec**2/2.
        Vy_mean = Vy_0
        Y_mean_vec = Y_0[None] + Vy_0[None]*Time_vec
        Vz_mean_vec = tf.sign(Vz_0[None]) * tf.sqrt(Vz_0[None]**2 -2.*Phi_Z_vec)
        Vxvyvz_mean_vec = tf.transpose(tf.stack([Vx_mean_vec, Vy_mean*tf.ones(tf.shape(Vx_mean_vec)), Vz_mean_vec])) -V_sun[None,:]
        Vlvbvr_mean_vec = tf.linalg.matvec(self.Rot_matrices, Vxvyvz_mean_vec)
        M_vec = tf.transpose( tf.stack([1./(4.74057*Dist_vec), 1./(4.74057*Dist_vec), tf.ones(tf.shape(Dist_vec), dtype=tf.float32)]) )
        Mlmbvr_mean_vec = Vlvbvr_mean_vec * M_vec
        Sigma_vxvyvz_vec = tf.transpose( tf.stack(      \
                            [ tf.stack( [Sigma_vx[None]**2*tf.ones(tf.shape(Vx_mean_vec), dtype=tf.float32), tf.zeros(tf.shape(Vx_mean_vec), dtype=tf.float32), tf.zeros(tf.shape(Vx_mean_vec), dtype=tf.float32)] ),  \
                              tf.stack( [tf.zeros(tf.shape(Vx_mean_vec), dtype=tf.float32), Sigma_vy[None]**2*tf.ones(tf.shape(Vx_mean_vec), dtype=tf.float32), tf.zeros(tf.shape(Vx_mean_vec), dtype=tf.float32)] ),  \
                              tf.stack( [tf.zeros(tf.shape(Vx_mean_vec), dtype=tf.float32), tf.zeros(tf.shape(Vx_mean_vec), dtype=tf.float32), Sigma_vz[None]**2*tf.ones(tf.shape(Vx_mean_vec), dtype=tf.float32)] )   \
                            ] ), [2,0,1] )
        Sigma_vlvbvr_vec = tf.linalg.matmul( tf.linalg.matmul(self.Rot_matrices, Sigma_vxvyvz_vec), self.Rot_matrices_inv)
        M_matrix_vec = tf.linalg.diag( M_vec )
        Sigma_mlmbvr_vec = tf.linalg.matmul( tf.linalg.matmul(M_matrix_vec, Sigma_vlvbvr_vec), M_matrix_vec)
        
        # this is for when v_RV is available
        S_mlmbvr = tf.gather(Sigma_mlmbvr_vec, self.Vlos_incl_indices) + tf.gather(self.Sigma_mlmbvr_obs_vec, self.Vlos_incl_indices)
        Diff_mlmbvr = tf.gather(self.Mlmbvr_obs_vec, self.Vlos_incl_indices)-tf.gather(Mlmbvr_mean_vec, self.Vlos_incl_indices)
        Mlmbvr_det_vec = S_mlmbvr[:,0,0] * S_mlmbvr[:,1,1] * S_mlmbvr[:,2,2] \
                       + S_mlmbvr[:,0,1] * S_mlmbvr[:,1,2] * S_mlmbvr[:,2,0] \
                       + S_mlmbvr[:,1,2] * S_mlmbvr[:,0,2] * S_mlmbvr[:,0,1] \
                       - S_mlmbvr[:,1,1] * S_mlmbvr[:,0,2]**2 \
                       - S_mlmbvr[:,2,2] * S_mlmbvr[:,0,1]**2 \
                       - S_mlmbvr[:,0,0] * S_mlmbvr[:,1,2]**2
        S_mlmbvr_inv = tf.transpose( tf.stack([
            tf.stack([ S_mlmbvr[:,2,2]*S_mlmbvr[:,1,1]-S_mlmbvr[:,1,2]**2, \
                       S_mlmbvr[:,0,2]*S_mlmbvr[:,1,2]-S_mlmbvr[:,2,2]*S_mlmbvr[:,0,1], \
                       S_mlmbvr[:,0,1]*S_mlmbvr[:,1,2]-S_mlmbvr[:,0,2]*S_mlmbvr[:,1,1] ]),\
            tf.stack([ S_mlmbvr[:,0,2]*S_mlmbvr[:,1,2]-S_mlmbvr[:,2,2]*S_mlmbvr[:,0,1], \
                       S_mlmbvr[:,2,2]*S_mlmbvr[:,0,0]-S_mlmbvr[:,0,2]**2, \
                       S_mlmbvr[:,0,1]*S_mlmbvr[:,0,2]-S_mlmbvr[:,0,0]*S_mlmbvr[:,1,2] ] ),\
            tf.stack([ S_mlmbvr[:,0,1]*S_mlmbvr[:,1,2]-S_mlmbvr[:,0,2]*S_mlmbvr[:,1,1], \
                       S_mlmbvr[:,0,1]*S_mlmbvr[:,0,2]-S_mlmbvr[:,0,0]*S_mlmbvr[:,1,2], \
                       S_mlmbvr[:,0,0]*S_mlmbvr[:,1,1]-S_mlmbvr[:,0,1]**2]),\
            ]), [2,0,1]) / Mlmbvr_det_vec[:,None,None]
        Log_velocity_prob_vec_vlos_incl = -1./2.*tf.reduce_sum( Diff_mlmbvr * tf.linalg.matvec(S_mlmbvr_inv, Diff_mlmbvr), [1]) \
                                          -1./2.*tf.math.log((2.*pi)**3.*Mlmbvr_det_vec)
        
        # this is for when v_RV is missing
        S_mlmbvr = tf.gather(Sigma_mlmbvr_vec[:,0:2,0:2], self.Vlos_excl_indices) + tf.gather(self.Sigma_mlmbvr_obs_vec[:,0:2,0:2], self.Vlos_excl_indices)
        Diff_mlmbvr = tf.gather(self.Mlmbvr_obs_vec[:,0:2], self.Vlos_excl_indices) - tf.gather(Mlmbvr_mean_vec[:,0:2], self.Vlos_excl_indices)
        Mlmbvr_det_vec = S_mlmbvr[:,0,0]*S_mlmbvr[:,1,1] -S_mlmbvr[:,0,1]*S_mlmbvr[:,1,0]
        S_mlmbvr_inv = tf.transpose( tf.stack( [ \
            tf.stack( [ S_mlmbvr[:,1,1], -S_mlmbvr[:,0,1]] ), \
            tf.stack( [-S_mlmbvr[:,1,0],  S_mlmbvr[:,0,0]] )  \
            ] ), [2,0,1]) / Mlmbvr_det_vec[:,None,None]
        Log_velocity_prob_vec_vlos_excl = -1./2.*tf.reduce_sum( Diff_mlmbvr * tf.linalg.matvec(S_mlmbvr_inv, Diff_mlmbvr), [1]) \
                                          -1./2.*tf.math.log((2.*pi)**2.*Mlmbvr_det_vec)
        
        # spatial part
        Log_pos_prob_vec = -( X_vec-X_mean_vec )**2 / ( 2.*Sigma_x[None]**2. ) - tf.math.log((2.*pi)**0.5*Sigma_x) + \
                           -( Y_vec-Y_mean_vec )**2 / ( 2.*Sigma_y[None]**2. ) - tf.math.log((2.*pi)**0.5*Sigma_y)
        Log_par_prob_vec = -1./2. * (Par_vec-self.Par_obs_vec)**2/self.Par_err_vec**2 + 2.*tf.math.log(Dist_vec) + Log_nuisance_jacobian
        Log_spatial_prob_vec = Log_pos_prob_vec + Log_par_prob_vec
        
        # add it all together
        Log_post_vec_vlos_incl = Log_velocity_prob_vec_vlos_incl + tf.gather(Log_spatial_prob_vec, self.Vlos_incl_indices)
        Log_post_vec_vlos_excl = Log_velocity_prob_vec_vlos_excl + tf.gather(Log_spatial_prob_vec, self.Vlos_excl_indices)
        
        # prior
        Log_prior = -(Z_sun-self.Z_sun_true)**2./(2.*0.005**2.) \
                -(Vx_sun-self.Vx_sun_true)**2./(2.*5.**2.) \
                -(Vy_sun-self.Vy_sun_true)**2./(2.*5.**2.) \
                -(Vz_sun-self.Vz_sun_true)**2./(2.*0.1**2.) \
                -(Fx-self.Fx_true)**2./(2.*(0.025*self.Fx_true)**2.) \
                -(1e-2/Sigma_vx)**4 -(1e-2/Sigma_vy)**4 -(1e-2/Sigma_vz)**4 \
                -(1e-4/Sigma_x)**4 -(1e-4/Sigma_y)**4
        Ln_post = Log_prior + tf.reduce_sum(Log_post_vec_vlos_incl) + tf.reduce_sum(Log_post_vec_vlos_excl)
        return Ln_post
    
    
    @tf.function
    def lnpost_of_Vector_with_npp(self, Vector):
        Params = self.Vector_2_Params(Vector[0:16])
        Nuisance_params = Vector[16:16+self.obs_list_len]
        LogPost = self.unnormalized_posterior_log_prob_with_nuisance_par_params(Params, Nuisance_params)
        LogCoordTransPrior = tf.reduce_sum( tf.math.log( (self.Var_bounds[:,1]-self.Var_bounds[:,0])*tf.exp(-Vector[0:16])/(tf.exp(-Vector[0:16])+1.)**2 ) )
        return LogPost + LogCoordTransPrior
    
    
    def get_ideal_trajectory(self, p0, zlims=[-0.4,0.4]):
        Params = tf.Variable(p0, dtype=tf.float32)
        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        #sess = tf.compat.v1.Session(config=session_conf)
        with tf.compat.v1.Session(config=session_conf) as sess:
            sess.run( tf.compat.v1.global_variables_initializer() )
            res = sess.run(self.ideal_trajectory(Params, zlims=zlims))
        return res
    
    
    def get_lnpost_value(self, p0):
        Params = tf.Variable(p0, dtype=tf.float32)
        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        #sess = tf.compat.v1.Session(config=session_conf)
        with tf.compat.v1.Session(config=session_conf) as sess:
            sess.run( tf.compat.v1.global_variables_initializer() )
            res = sess.run(self.unnormalized_posterior_log_prob_with_nuisance_par_params(Params[0:16],Params[16:16+len(self.obs_list)]))
        return res
    
    
    # minimize the posterior using and optimizer
    # note that this function includes the distance nuisance parameters!!
    def minimize_posterior(self, p0, number_of_iterations=10000, print_gap=500, plot_gap=None, numTFTthreads=1, learning_rate=1e-3):
        vector = p0_2_vector(p0[0:16])
        Vector = tf.Variable(vector, dtype=tf.float32)
        Params = self.Vector_2_Params(Vector)
        Nuisance_params = tf.Variable(p0[16:len(p0)], dtype=tf.float32)
        MinusLogPosterior = -self.unnormalized_posterior_log_prob_with_nuisance_par_params(Params, Nuisance_params)
        IdealTrajectory = self.ideal_trajectory(Params)
        Optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(MinusLogPosterior)
        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=numTFThreads, inter_op_parallelism_threads=numTFThreads)
        with tf.compat.v1.Session(config=session_conf) as sess:
            sess.run( tf.compat.v1.global_variables_initializer() )
            for i in range(number_of_iterations):
                _, minusLogPosterior, params, nuisance_params  =  \
                    sess.run([Optimizer, MinusLogPosterior, Params, Nuisance_params])
                if np.isnan(minusLogPosterior):
                    print(minusLogPosterior)
                    print(list(params), "\n\n")
                if i%print_gap==0:
                    print(minusLogPosterior)
                    print(list(params), "\n\n")
                if plot_gap!=None and i%plot_gap==0 and i>0:
                    obs_ideal = sess.run(IdealTrajectory)
                    df_A = pd.DataFrame(np.c_[self.obs_list, np.zeros(np.shape(self.obs_list)[0])], columns=['par','l','b','mul','mub','vlos', 'info'])
                    df_B = pd.DataFrame(np.c_[obs_ideal, np.ones(np.shape(obs_ideal)[0])], columns=['par','l','b','mul','mub','vlos', 'info'])
                    df_join = pd.concat([df_A, df_B])
                    g = sns.pairplot(df_join, height=2., vars=['par','l','b','mul','mub','vlos'], markers=".", hue='info')
                    plt.show()
        return params, nuisance_params
    
    
    # run the Hamiltonian Monte-Carlo
    def run_HMC(self, p0, steps=1e5, burnin_steps=0, num_adaptation_steps=0, num_leapfrog_steps=3, \
                step_size_start=np.array([1e-3,1e-6,1e-3,1e-4,1e-5,1e-5,1e-5,1e-2,1e-2,1e-2,1e-2,1e-2,1e-3,1e-4,1e-5,1e-5])):
        tf.compat.v1.disable_eager_execution()
        v0 = np.concatenate( (p0_2_vector(p0[0:16]), p0[16:len(p0)]) )
        the_function = self.lnpost_of_Vector_with_npp
        num_results = int(steps)
        num_burnin_steps = int(burnin_steps)
        adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=the_function,
                num_leapfrog_steps=num_leapfrog_steps,
                step_size=step_size_start),
            num_adaptation_steps=int(num_adaptation_steps))
        @tf.function
        def run_chain():
            # Run the chain (with burn-in).
            samples, [is_accepted, step_size, log_prob] = tfp.mcmc.sample_chain(
                num_results=num_results,
                num_burnin_steps=num_burnin_steps,
                current_state=np.array(v0, dtype=np.float32),
                kernel=adaptive_hmc,
                trace_fn=lambda _, pkr: [pkr.inner_results.is_accepted, pkr.inner_results.accepted_results.step_size, pkr.inner_results.accepted_results.target_log_prob])
            is_accepted = tf.reduce_mean(tf.cast(is_accepted, dtype=tf.float32))
            return samples, is_accepted, step_size, log_prob
        Samples, Is_accepted, Step_size, Log_prob = run_chain()
        with tf.compat.v1.Session() as sess:
            samples, is_accepted, step_size, log_prob = sess.run([Samples, Is_accepted, Step_size, Log_prob])
        param_samples = v_2_params(samples[:,0:16])
        return param_samples, samples, log_prob, step_size, is_accepted
    
    
    # burn-in the Hamiltonian Monte-Carlo
    def burnin_with_npp(self, p0, step_size_start, steps=1e4, burnin_steps=1e3, num_adaptation_steps=1e3, num_leapfrog_steps=1, iterations=1):
        pN = p0
        ssN = step_size_start
        for i in range(iterations):
            if iterations>1:
                print('Burnin iteration:',i+1,'/',iterations)
            param_samples, samples, log_prob, step_size, is_accepted = self.run_HMC(pN, steps=steps, burnin_steps=burnin_steps, num_adaptation_steps=num_adaptation_steps, num_leapfrog_steps=num_leapfrog_steps, step_size_start=ssN)
            pN = np.concatenate( (param_samples[-1],samples[-1,16:16+len(self.obs_list)]) )
            ssN = 1e-1 * np.sqrt( [np.cov(ss) for ss in np.transpose(samples)] )
            print(np.median(param_samples, axis=0))
            print(np.sqrt( [np.cov(ss) for ss in np.transpose(param_samples)][0:16] ))
        return pN, ssN
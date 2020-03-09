import astropy.units as u
import numpy as np
import gala.potential as gp
import gala.dynamics as gd
from gala.dynamics.mockstream import mock_stream
from gala.units import galactic
import gala.coordinates as gc
import astropy.coordinates as coord
import matplotlib.pyplot as plt
from numpy import *
from math import *
from numpy.linalg import inv
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import operator
import sys


from gala.potential import MilkyWayPotential
import gala.potential as gp
from gala.potential import load

# https://gala-astro.readthedocs.io/en/latest/examples/mock-stream-heliocentric.html


# make a stream orbit, randomise stars along it, and add phase-space dispersion to them
def make_perfect_stream(X_start=[-8.122, 0., 0.5], V_start=[-16., 0., -200.], dt=.00048, n_steps=10000, number_of_stars=1000, intrinsic_spread=None):
    pot=MilkyWayPotential()
    kpcMyr2kms = 977.951/(u.kpc/u.Myr)
    w0 = gd.PhaseSpacePosition(pos=X_start*u.kpc, vel=V_start*u.km/u.s)
    orbit = gp.Hamiltonian(pot).integrate_orbit(w0, dt=dt, n_steps=n_steps)
    indices = np.random.randint(0,len(orbit.x),size=number_of_stars)
    orb_x = orbit.x[indices]/u.kpc
    orb_y = orbit.y[indices]/u.kpc
    orb_z = orbit.z[indices]/u.kpc
    orb_vx = orbit.v_x[indices]*kpcMyr2kms
    orb_vy = orbit.v_y[indices]*kpcMyr2kms
    orb_vz = orbit.v_z[indices]*kpcMyr2kms
    if intrinsic_spread!=None:
        for i in range(len(orb_x)):
            z_pot_start = pot.value([orb_x[i],orb_y[i],orb_z[i]])[0]*978.5**2/(u.kpc/u.Myr)**2
            orb_z[i] += intrinsic_spread[0]*np.random.normal()
            z_pot_diff = pot.value([orb_x[i],orb_y[i],orb_z[i]])[0]*978.5**2/(u.kpc/u.Myr)**2-z_pot_start
            orb_x[i] += intrinsic_spread[0]*np.random.normal()
            orb_y[i] += intrinsic_spread[0]*np.random.normal()
            orb_vx[i] += intrinsic_spread[1]*np.random.normal()
            orb_vy[i] += intrinsic_spread[1]*np.random.normal()
            # this last line adds dispersion to v_z, but in a way that counteracts
            # the change in vertical energy coming from the shift in height z
            orb_vz[i] = np.sign(orb_vz[i])*np.sqrt( orb_vz[i]**2 - 2.*z_pot_diff )+intrinsic_spread[1]*np.random.normal()
    return np.transpose([orb_x,orb_y,orb_z,orb_vx,orb_vy,orb_vz])



# EXAMPLES:

# this is stream S1
# xyzs = make_perfect_stream(X_start=[-8.122,0., 0.4], V_start=[-25., 0., -100.], dt=2.*2.*.00048, intrinsic_spread=[0.020, 1.], n_steps=4000)
# np.savez('../GeneratedStreams/mock_stream_20-1000_slow', xyzs=xyzs)

# this is stream S2
# xyzs = make_perfect_stream(X_start=[-8.122, -.6, -0.2], V_start=[0., 220., -50.], dt=2.*2.*.00048, intrinsic_spread=[0.020, 1.], n_steps=2700)
# np.savez('../GeneratedStreams/mock_stream_20-1000_corotB', xyzs=xyzs)

# this is stream S3
# xyzs = make_perfect_stream(X_start=[-8.122, -.3, 0.4], V_start=[-10., 200., -70.], dt=2.*.00048, intrinsic_spread=[0.020, 1.], n_steps=2500)
# np.savez('../GeneratedStreams/mock_stream_20-1000_corotC', xyzs=xyzs)

# this is stream S4
# xyzs = make_perfect_stream(X_start=[-7.835, -.3, 0.4], V_start=[-160., 160., -60.], dt=2.*.00048, intrinsic_spread=[0.020, 1.], n_steps=3800)
# np.savez('../GeneratedStreams/mock_stream_20-1000_inclined', xyzs=xyzs)
from firedrake import *
from nudging import *
import numpy as np
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc
from pyadjoint import AdjFloat

from nudging.models.stochastic_Camassa_Holm import Camsholm

""" read obs from saved file 
    Do assimilation step for tempering and jittering steps 
"""

nsteps = 5
xpoints = 40
model = Camsholm(100, nsteps, xpoints, lambdas=True)
MALA = True
verbose = True
nudging = True
jtfilter = jittertemp_filter(n_jitt = 4, delta = 0.1,
                             verbose=verbose, MALA=MALA,
                             nudging=True, visualise_tape=True)

nensemble = [5,5,5,5]
jtfilter.setup(nensemble, model)

x, = SpatialCoordinate(model.mesh) 

#prepare the initial ensemble
for i in range(nensemble[jtfilter.ensemble_rank]):
    dx0 = model.rg.normal(model.R, 0., 0.05)
    dx1 = model.rg.normal(model.R, 0., 0.05)
    a = model.rg.uniform(model.R, 0., 1.0)
    b = model.rg.uniform(model.R, 0., 1.0)
    u0_exp = (1+a)*0.2*2/(exp(x-403./15. + dx0) + exp(-x+403./15. + dx0)) \
        + (1+b)*0.5*2/(exp(x-203./15. + dx1)+exp(-x+203./15. + dx1))

    _, u = jtfilter.ensemble[i][0].split()
    u.interpolate(u0_exp)

def log_likelihood(y, Y):
    ll = (y-Y)**2/0.05**2/2*dx
    return ll

#Load data
y_exact = np.load('y_true.npy')
y = np.load('y_obs.npy') 
N_obs = y.shape[0]

yVOM = Function(model.VVOM)

# do assimiliation step
for k in range(N_obs):
    PETSc.Sys.Print("Step", k)
    yVOM.dat.data[:] = y[k, :]

    # actually do the data assimilation step
    jtfilter.assimilation_step(yVOM, log_likelihood)

if COMM_WORLD.rank == 0:
    print("Time shape", y_sim_obs_alltime_step.shape)
    print("Obs shape", y_sim_obs_allobs_step.shape)
    print("Ensemble member", y_e.shape)
    np.save("assimilated_ensemble.npy", y_e)
    np.save("simualated_all_time_obs.npy", y_sim_obs_allobs_step)

"""
Debugging notes
replaced R space with DG 0, still issue
"""

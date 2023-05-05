from firedrake import *
from nudging import *
import numpy as np
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc
from pyadjoint import AdjFloat

from stochastic_Camassa_Holm import Camsholm1 as Camsholm

""" read obs from saved file 
    Do assimilation step for tempering and jittering steps 
"""

nsteps = 5
model = Camsholm(100, nsteps)
MALA = False
verbose = True
jtfilter = jittertemp_filter(n_temp=4, n_jitt = 4, rho= 0.4,
                            verbose=verbose, MALA=MALA)


nensemble = [5] * 20

jtfilter.setup(nensemble, model)

x, = SpatialCoordinate(model.mesh) 

for i in range(nensemble[jtfilter.ensemble_rank]):
    dx0 = model.rg.normal(model.R, 0., 2.0)
    dx1 = model.rg.normal(model.R, 0., 2.0)
    a = model.rg.uniform(model.R, 0.5, 1.5)
    b = model.rg.uniform(model.R, 0.5, 1.5)
    u0_exp = a*0.2*2/(exp(x-403./15. - dx0) + exp(-x+403./15. + dx0)) \
        + b*0.5*2/(exp(x-203./15. - dx1)+exp(-x+203./15. + dx1))
    _, u = jtfilter.ensemble[i][0].subfunctions
    u.interpolate(u0_exp)

def log_likelihood(y, Y):
    ll = (y-Y)**2/0.05**2/2*dx
    return ll
    
#Load data
y_exact = np.loadtxt('y_true.dat')
y = np.loadtxt('y_obs.dat') 
N_obs = y.shape[0]

yVOM = Function(model.VVOM)

# prepare shared arrays for data
y_e_list = []
for m in range(y.shape[1]):        
    y_e_shared = SharedArray(partition=nensemble, 
                                 comm=jtfilter.subcommunicators.ensemble_comm)
    y_e_list.append(y_e_shared)

ys = y.shape
if COMM_WORLD.rank == 0:
    y_e = np.zeros((np.sum(nensemble), ys[1]+1))

# initial data plots
for i in range(nensemble[jtfilter.ensemble_rank]):
    model.w0.assign(jtfilter.ensemble[i][0])
    obsdata = model.obs().dat.data[:]
    for m in range(y.shape[1]):
        y_e_list[m].dlocal[i] = obsdata[m]

for m in range(y.shape[1]):
    y_e_list[m].synchronise()
    if COMM_WORLD.rank == 0:
        y_e[:, m] = y_e_list[m].data()

if COMM_WORLD.rank == 0:
    np.savetxt("ensemble_simulated_obs0.txt", y_e)

        
# do assimiliation step
for k in range(y.shape[0]):
    PETSc.Sys.Print("Step", k)
    yVOM.dat.data[:] = y[k, :]
    jtfilter.assimilation_step(yVOM, log_likelihood)
        
    for i in range(nensemble[jtfilter.ensemble_rank]):
        model.w0.assign(jtfilter.ensemble[i][0])
        obsdata = model.obs().dat.data[:]
        for m in range(y.shape[1]):
            y_e_list[m].dlocal[i] = obsdata[m]

    for m in range(y.shape[1]):
        y_e_list[m].synchronise()
        if COMM_WORLD.rank == 0:
            y_e[:, m] = y_e_list[m].data()

    if COMM_WORLD.rank == 0:
        np.savetxt("ensemble_simulated_obs"+str(k+1)+".txt", y_e)

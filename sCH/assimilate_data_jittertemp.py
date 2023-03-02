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
model = Camsholm(100, nsteps)
MALA = True
verbose = True
jtfilter = jittertemp_filter(n_temp=4, n_jitt = 4, rho= 0.4,
                            verbose=verbose, MALA=MALA)


nensemble = [5] * 20

jtfilter.setup(nensemble, model)

x, = SpatialCoordinate(model.mesh) 

# elliptic problem to have smoother initial conditions in space
p = TestFunction(model.V)
q = TrialFunction(model.V)
xi = Function(model.V)
a = inner(grad(p), grad(q))*dx + p*q*dx
L = p*xi*dx
dW = Function(model.V)
dW_prob = LinearVariationalProblem(a, L, dW)
dw_solver = LinearVariationalSolver(dW_prob,
                                         solver_parameters={'mat_type': 'aij', 'ksp_type': 'preonly','pc_type': 'lu'})
for i in range(nensemble[jtfilter.ensemble_rank]):
    xi.assign(model.rg.uniform(model.V, 0., 1.0))
    dw_solver.solve()
    _, u = jtfilter.ensemble[i][0].split()
    u.assign(dW)

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
    y_e = np.zeros((np.sum(nensemble), ys[0], ys[1]))

# do assimiliation step
for k in range(2):
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
            y_e[:, k, m] = y_e_list[m].data()

if COMM_WORLD.rank == 0:
    np.savetxt("ensemble_simulated_obs.txt", y_e)

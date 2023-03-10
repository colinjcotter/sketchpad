from ctypes import sizeof
from fileinput import filename
from firedrake import *
from pyop2.mpi import MPI
from nudging import *
import numpy as np
from stochastic_Camassa_Holm import Camsholm1 as Camsholm

"""
create some synthetic data/observation data at T_1 ---- T_Nobs
Pick initial conditon
run model, get obseravation
add observation noise N(0, sigma^2)
"""
nsteps = 5
model = Camsholm(100, nsteps, dt=0.1)
model.setup()
X_truth = model.allocate()
_, u0 = X_truth[0].split()
x, = SpatialCoordinate(model.mesh)
u0.interpolate(0.2*2/(exp(x-403./15.) + exp(-x+403./15.)) + 0.5*2/(exp(x-203./15.)+exp(-x+203./15.)))

N_obs = 50

y_true_VOM = model.obs()
y_true = y_true_VOM.dat.data[:]
y_obs_full = np.zeros((N_obs, np.size(y_true)))
y_true_full = np.zeros((N_obs, np.size(y_true)))

for i in range(N_obs):
    print("step", i)
    model.randomize(X_truth)
    model.run(X_truth, X_truth)
    y_true_VOM = model.obs()
    y_true = y_true_VOM.dat.data[:]

    y_true_full[i,:] = y_true
    y_true_data = np.savetxt("y_true.dat", y_true_full)

    y_noise = np.random.normal(0.0, 0.05, 40)  

    y_obs = y_true + y_noise
    
    y_obs_full[i,:] = y_obs 

# need to save into rank 0
np.savetxt("y_obs.dat", y_obs_full)

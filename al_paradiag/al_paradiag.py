from firedrake import *

n = 20
m = UnitIntervalMesh(n)

ntime = 20
Tmax = 1.0
dt = Tmax/ntime

degree = 1
V = VectorFunctionSpace(mesh, "CG", degree, dim=ntime) # velocity space
Q = VectorFunctionSpace(mesh, "DG", degree-1, dim=ntime)

W = V * Q



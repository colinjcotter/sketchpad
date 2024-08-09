from firedrake import *
from math import pi, ceil

ncells = 100
L = 10
mesh = PeriodicIntervalMesh(ncells, L)

V = FunctionSpace(mesh, "Hermite", 3)

un = Function(V)
unp1 = Function(V)

uh = (un + unp1)/2

v = TestFunction(V)

dt = 0.01
dT = Constant(dt)

# stochastic bits

DG0 = FunctionSpace(mesh, "DG", 0)
pcg = PCG64(seed=84574584563)
rg = RandomGenerator(pcg)
dW = Function(DG0)

alpha = Constant(1.0) # viscosity
beta = Constant(0.02923) # hyperviscosity
gamma = Constant(1.) # advection
dc = Constant(0.001) # diffusion coefficient for noise

eqn = (
    v*(unp1 - un)*dx
    - dT*alpha*v.dx(0)*uh.dx(0)*dx
    + dT*beta*(
        v.dx(0).dx(0)*uh.dx(0).dx(0)*dx
               )
    - dT*gamma*0.5*v.dx(0)*uh*uh*dx
    - dc*dW*v*dx
    )

params = {
    "snes_atol": 1.0e-50,
    "snes_rtol": 1.0e-6,
    "snes_stol": 1.0e-50,
    "ksp_type":"preonly",
    "pc_type":"lu"
}

#make the solver
KSProb = NonlinearVariationalProblem(eqn, unp1)
KSSolver = NonlinearVariationalSolver(KSProb,
                                      solver_parameters=params)

#initial condition

x, = SpatialCoordinate(mesh)

t = 0.
tmax = 1.

area = CellVolume(mesh)
nsteps = ceil(tmax/dt)


VOM = VertexOnlyMesh(mesh, [[3.43]])
VVOM = FunctionSpace(VOM, "DG", 0)
CG3 = FunctionSpace(mesh, "CG", 3)
uout = Function(CG3)
vals = []

nsamples = 100
for sample in range(nsamples):
    print(sample)
    un.project(sin(pi*2*x) + 0.2*cos(pi*x))
    for step in ProgressBar("timestep").iter(range(nsteps)):
        t += dt
        
        new_dW = rg.normal(DG0, 0. , 1.)
        dW.interpolate(dt**0.5*new_dW/area**0.5)
        KSSolver.solve()
        un.assign(unp1)
    uout.interpolate(un)
    y = assemble(interpolate(uout, VVOM))
    vals.append(y.dat.data[0])

import matplotlib.pyplot as pp
pp.hist(vals)
pp.show()

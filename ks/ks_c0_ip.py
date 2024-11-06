from firedrake import *
from math import pi

ncells = 100
L = 10
mesh = PeriodicIntervalMesh(ncells, L)

V = FunctionSpace(mesh, "CG", 2)
Vdg = FunctionSpace(mesh, "DG", 1)
un = Function(V)
unp1 = Function(V)

uh = (un + unp1)/2

v = TestFunction(V)

dt = 0.01
dT = Constant(dt)

alpha = Constant(1.0) # viscosity
beta = Constant(0.02923) # hyperviscosity
gamma = Constant(1.) # advection

eta = Constant(5.)

def a(u, v):
    h = avg(CellVolume(mesh))/FacetArea(mesh)
    eqn = v.dx(0).dx(0)*u.dx(0).dx(0)*dx
    eqn += avg(u.dx(0).dx(0))*jump(v.dx(0))*dS
    eqn += avg(v.dx(0).dx(0))*jump(u.dx(0))*dS
    eqn += eta/h*jump(v.dx(0))*jump(u.dx(0))*dS
    return eqn

eqn = (
    v*(unp1 - un)*dx
    - dT*alpha*v.dx(0)*uh.dx(0)*dx
    + a(dT*beta*uh, v)
    - dT*gamma*0.5*v.dx(0)*uh*uh*dx
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
un.project(exp(sin(pi*2*x) + 0.2*cos(pi*x)))

t = 0.
tmax = 100.
tdump = 0.1
dumpt = 0.

file0 = File("stuff.pvd")
uout = Function(Vdg)
uout.interpolate(un)
file0.write(uout)

while t < tmax - dt/2:
    t += dt

    KSSolver.solve()
    print(norm(unp1))
    un.assign(unp1)

    if dumpt > tdump - dt/2:
        uout.interpolate(un)
        file0.write(uout)
        dumpt -= tdump
    dumpt += dt

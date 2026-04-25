from firedrake import *
from irksome import Dt, GaussLegendre, TimeStepper

alpha = 1.0
alphasq = Constant(alpha**2)
dt = 0.1

n = 100
mesh = PeriodicIntervalMesh(n, 40.0)

V = FunctionSpace(mesh, "CG", 1)
W = V * V * V

w0 = Function(W)
mhat0, dHdmhat0, u0 = w0.subfunctions

x, = SpatialCoordinate(mesh)
uplot = Function(V)
m0 = Function(V)
p = TestFunction(V)
m = TrialFunction(V)

m0.interpolate(0.2*2/(exp(x-403./15.) + exp(-x+403./15.))
                  + 0.5*2/(exp(x-203./15.)+exp(-x+203./15.)))

# solver to plot u
u = TrialFunction(V)
am = (p*u + alphasq*p.dx(0)*u.dx(0))*dx
Lm = p*mhat0**2*dx

uplot = Function(V)
uprob = LinearVariationalProblem(am, Lm, uplot)
usolver = LinearVariationalSolver(uprob)

mhat0.interpolate(m0**0.5)

#mhat0, IdHdmhat0, Iu0
p, q, r = TestFunctions(W)

mhat0, IdHdmhat0, Iu0 = split(w0)
dHdmhat0 = Dt(IdHdmhat0)
u0 = Dt(Iu0)

F = (
   (r*u0 + alphasq*r.dx(0)*u0.dx(0) - r*mhat0**2)*dx +
   q*(dHdmhat0 - mhat0*u0)*dx 
   + (p*Dt(mhat0) + 0.5*(p*dHdmhat0.dx(0) - p.dx(0)*dHdmhat0))*dx
)

t = 0.0
tc = Constant(t)
dtc = Constant(dt)

butcher_tableau = GaussLegendre(1)
luparams = {"mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu"}

stepper = TimeStepper(F, butcher_tableau, tc, dtc, w0,
                      solver_parameters=luparams)

T = 1000.0
ufile = VTKFile('sqrtch.pvd')
usolver.solve()
ufile.write(uplot, time=t)
all_us = []

ndump = 50
dumpn = 0

#FOR VISUALISATION WILL NEED TO SOLVE FOR u0!

energy = []

while (t < T - 0.5*dt):
   t += dt
   usolver.solve()
   u = uplot
   E = assemble((u*u + alphasq*u.dx(0)*u.dx(0))*dx)
   energy.append(E)
   print("t = ", t, "E = ", E)
   stepper.advance()

   dumpn += 1
   if dumpn == ndump:
      dumpn -= ndump
      ufile.write(uplot, time=t)

from numpy import *
import matplotlib.pyplot as pp
t = arange(len(energy))*dt
pp.plot(t, energy)
pp.xlabel("Time")
pp.ylabel("Energy")
pp.show()

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
mhat0, _, _ = w0.subfunctions

x, = SpatialCoordinate(mesh)
u0 = Function(V).interpolate(0.1*(exp(-(x-10.0)**2/2)
                                  -exp(-(x-30.0)**2/2)))
uplot = Function(V)

#solver to get m0
m0 = Function(V)
p = TestFunction(V)
m = TrialFunction(V)
am = inner(m,p)*dx
Lm = (inner(u0,p)
      + alphasq*inner(u0.dx(0),p.dx(0))
      )*dx
solve(am == Lm, m0)
mhat0.interpolate(sqrt(m0))

# solver to plot u
u = TrialFunction(V)
am = (inner(u,p)
      + alphasq*inner(u.dx(0),p.dx(0))
      )*dx
Lm = inner(mhat0**2,p)*dx
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
   (inner(u0,r) + alphasq*inner(u0.dx(0),r.dx(0)) - inner(mhat0**2,r))*dx +
   inner(dHdmhat0 - mhat0*u0,q)*dx 
   + (inner(Dt(mhat0),p) + 0.5*(inner(dHdmhat0.dx(0),p)
                                - inner(dHdmhat0,p.dx(0))))*dx
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

ndump = 50
dumpn = 0

energy = []

while (t < T - 0.5*dt):
   t += dt
   usolver.solve()
   u = uplot
   E = assemble((inner(u,u) + alphasq*inner(u.dx(0),u.dx(0)))*dx)
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

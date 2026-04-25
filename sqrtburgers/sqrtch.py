from firedrake import *
from irksome import *

alpha = 1.0
alphasq = Constant(alpha**2)
dt = 0.1
Dt = Constant(dt)

n = 100
mesh = PeriodicIntervalMesh(n, 40.0)

V = FunctionSpace(mesh, "CG", 1)
W = V * V * V

w0 = Function(W)
mhat0, dHdmhat0, u0 = w0.subfunctions

x, = SpatialCoordinate(mesh)
u0.interpolate(0.2*2/(exp(x-403./15.) + exp(-x+403./15.))
               + 0.5*2/(exp(x-203./15.)+exp(-x+203./15.)))

m0 = Function(V)
p = TestFunction(V)
m = TrialFunction(V)

am = p*m*dx
Lm = (p*u0 + alphasq*p.dx(0)*u0.dx(0))*dx

solve(am == Lm, m0, solver_parameters={
      'ksp_type': 'preonly',
      'pc_type': 'lu'
      }
   )

mhat0 = Function(V).interpolate(m0**0.5)

#mhat0, IdHdmhat0, Iu0
p, q, r = TestFunctions(W)

mhat0, IdHdmhat0, Iu0 = split(w0)
dHdmhat0 = Dt(IdHdmhat0)
u0 = Dt(Iu0)

L = (
  (r*u0 + alphasq*r.dx(0)*u0.dx(0) - r*mhat0**2)*dx +
  q*(dHdmhat0 - mhat0*u0)*dx +
  (p*Dt(m0) + Dt*(p*uh.dx(0)*mh -p.dx(0)*uh*mh))*dx
)

T = 100.0
ufile = VTKFile('u.pvd')
t = 0.0
ufile.write(u0, time=t)
all_us = []

ndump = 10
dumpn = 0

while (t < T - 0.5*dt):
   t += dt
   E = assemble((u0*u0 + alphasq*u0.dx(0)*u0.dx(0))*dx)
   print("t = ", t, "E = ", E)

   usolver.solve()
   w0.assign(w1)

   dumpn += 1
   if dumpn == ndump:
      dumpn -= ndump
      ufile.write(u0, time=t)
      all_us.append(Function(u1))

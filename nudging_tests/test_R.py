from firedrake import *
from firedrake_adjoint import *

mesh = UnitIntervalMesh(20)

V = FunctionSpace(mesh, "R", 0)
W = V*V*V*V

a = Function(W).assign(1.0)

P = FunctionSpace(mesh, "CG", 1)

p = TrialFunction(P)
q = TestFunction(P)

p0 = Function(P)

solve(p*q*dx == q*(a.sub(0)+a.sub(1)+a.sub(2)+a.sub(3))*dx, p0)

Func = assemble(p0*p0*dx)

RFunc = ReducedFunctional(Func, [Control(a)])

g = RFunc.derivative()
print(norm(g[0]))

from firedrake import *
from firedrake_adjoint import *
pyadjoint.tape.pause_annotation()

mesh = UnitSquareMesh(50, 50)
V = FunctionSpace(mesh, "CG", 1)

u = Function(V)
f = Function(V)

x, y = SpatialCoordinate(mesh)

f.assign(cos(2*pi*x)*sin(2*pi*y))
v = TestFunction(V)
q = Function(V)
L = inner(grad(v), grad(u - q))*(1 + (u-q)**2)*dx - f

prob = NonlinearVariationalProblem(L, u)
solver = NonlinearVariationalSolver(prob, solver_parameters=
                                    {'ksp_type':'preonly',
                                     'pc_type':'lu'})

pyadjoint.tape.continue_annotation()
solver.solve()
J = assemble(u*u*dx)
Jhat = ReducedFunctional(J, [Control(q), Control(f)])
pyadjoint.tape.pause_annotation()

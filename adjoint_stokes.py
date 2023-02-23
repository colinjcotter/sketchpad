from firedrake import *
from firedrake_adjoint import *

#I think you want pyadjoint.tape.pause_annotation (link) and pyadjoint.tape.continue_annotation

mesh = UnitSquareMesh(10,10)
V = VectorFunctionSpace(mesh, "Lagrange", 2)
Q = FunctionSpace(mesh, "Lagrange", 1)
W = V*Q

v, q = TestFunctions(W)
u, p = TrialFunctions(W)

R = FunctionSpace(mesh, "R", 0)
F0 = Function(R) # control parameter
nu = Constant(1)     # Viscosity coefficient

x, y = SpatialCoordinate(mesh)
Forcing = as_vector([x*(1-x)*y*(1-y), 0])

noslip = DirichletBC(W.sub(0), (0, 0), (1,2,3))
static_bcs = [inflow]

a = nu*inner(grad(u), grad(v))*dx - inner(p, div(v))*dx - inner(q, div(u))*dx
L = inner(F0*Forcing, v)*dx

w = Function(W)
solve(a == L, w, bcs=bcs, solver_parameters={"mat_type": "aij",
                                             "ksp_type": "preonly",
                                             "pc_type": "lu",
                                             "pc_factor_shift_type": "inblocks"})

u, p = split(w)
alpha = Constant(10)

J = assemble(1./2*inner(grad(u), grad(u))*dx + alpha/2*F0*F0*dx)
m = Control(F0)
Jhat = ReducedFunctional(J, m)

get_working_tape().progress_bar = ProgressBar
g_opt = minimize(Jhat)


from firedrake import *
from firedrake_adjoint import *
import pyadjoint

#I think you want pyadjoint.tape.pause_annotation (link) and pyadjoint.tape.continue_annotation

pyadjoint.tape.pause_annotation()

mesh = UnitSquareMesh(10,10)
V = VectorFunctionSpace(mesh, "Lagrange", 2)
Q = FunctionSpace(mesh, "Lagrange", 1)
W = V*Q

v, q = TestFunctions(W)
u, p = TrialFunctions(W)


R = FunctionSpace(mesh, "R", 0)
F0 = Function(R) # control parameter
F0.assign(0.0)
nu = Constant(1)     # Viscosity coefficient
w = Function(W)

x, y = SpatialCoordinate(mesh)
Forcing = as_vector([x*(1-x)*y*(1-y), 0])

noslip = DirichletBC(W.sub(0), (0, 0), (1,2,3))
static_bcs = [noslip]

a = nu*inner(grad(u), grad(v))*dx - inner(p, div(v))*dx - inner(q, div(u))*dx
L = inner(F0*Forcing, v)*dx

pyadjoint.tape.continue_annotation()
solve(a == L, w, bcs=static_bcs, solver_parameters={"mat_type": "aij",
                                             "ksp_type": "preonly",
                                             "pc_type": "lu",
                                             "pc_factor_shift_type": "inblocks"})

u, p = split(w)
alpha = Constant(10)

J = assemble(1./2*inner(grad(u), grad(u))**2*dx + alpha/2*F0*F0*dx)
pyadjoint.tape.pause_annotation()

get_working_tape().progress_bar = ProgressBar
m = Control(F0)
Fnew = Function(R) # control parameter
Fnew.assign(2)
m.update(Fnew)
Jhat = ReducedFunctional(J, m)
g = Jhat.derivative()
print(g.dat.data[:])

F0.assign(2.0)
m = Control(F0)
Fnew.assign(3)
m.update(Fnew)
Jhat = ReducedFunctional(J, m)
g = Jhat.derivative()
print(g.dat.data[:])



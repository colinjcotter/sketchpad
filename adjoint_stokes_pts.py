from firedrake import *
from firedrake_adjoint import *
import pyadjoint

# a test with point evaluation

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

alpha = Constant(0.)

VMesh = VertexOnlyMesh(mesh, [[0.5,0.5], [0.6,0.5]])
VOV = FunctionSpace(VMesh, "DG", 0)

y = Function(VOV)
y.dat.data[:] = [0.1, 0.2]

Y = Function(VOV)
u, p = w.split()
Y.interpolate(u[0])

J1form = alpha/2*F0*F0*dx
J2form = (y - Y)**2*dx
J = assemble(J1form) + assemble(J2form)
m = Control(F0)
Jhat = ReducedFunctional(J, m)
pyadjoint.tape.pause_annotation()

get_working_tape().progress_bar = ProgressBar

F0.assign(2.)
Jhat(F0)
g = Jhat.derivative()
print(g.dat.data[:])

F0.assign(3.)
Jhat(F0)
g = Jhat.derivative()
print(g.dat.data[:])



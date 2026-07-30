from firedrake import *
from irksome import *

nx = 100
L = 100000
mesh = SquareMesh(nx, nx, L)

V = FunctionSpace(mesh, "BDM", 2)
Q = FunctionSpace(mesh, "DG", 1)

# v, Iu, G, D, Iu is the time integral of u
W = V * V * V * Q

# model parameters
f = Constant(1e-4)
g = Constant(10)
H = Constant(1000)
alpha = Constant(1000)

n = FacetNormal(mesh)

# topography
b = Function(Q).interpolate(0.)

U = Function(W)

# initial conditions
_, _, _, D = U.subfunctions
D.interpolate(H-b)

# equation system
dv, du, dG, dD = TestFunctions(W)
v, Iu, G, D = split(U)
u = Dt(Iu)

F = Dt(G)
ubar = F/D

def both(u):
    return 2*avg(u)

def perp(vec):
    return as_vector([-vec[1], vec[0]])

Upwind = 0.5 * (sign(dot(u, n)) + 1)

# v_t equation
eqn = inner(dv, Dt(v))*dx
eqn -= inner(perp(grad(inner(dv, perp(ubar)))), v)*dx
eqn += inner(both(perp(n)*inner(dv, perp(ubar))), both(Upwind*v))*dS
eqn += inner(dv, f*perp(ubar))*dx
eqn -= div(dv)*(
    inner(u, u)/2
    + g*(D+b)    
)*dx
eqn -= div(dv)*(2/3)*alpha**2*div(F)*div(F)*dx

# v-u relation
eqn += inner(v-u, du)*dx - (2/3)*alpha**2*div(du)*div(F)/D*dx

# F definition
eqn += inner(F - D*u, dG)*dx

# D transport
eqn += dD*(Dt(D) + div(F))*dx

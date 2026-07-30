from firedrake import *
import irksome

nx = 100
L = 100000
mesh = PeriodicSquareMesh(nx, nx, L)

V = FunctionSpace(mesh, "BDM", 2)
Q = FunctionSpace(mesh, "DG", 1)

# v, Iu, G, D, Iu is the time integral of u
W = V * V * V * Q

# model parameters
f = Constant(1e-4)
g = Constant(10)
H = Constant(1000)
alpha = Constant(0)
dt = 100.

n = FacetNormal(mesh)

# topography
b = Function(Q).interpolate(0.)

U = Function(W)

# initial conditions
_, _, _, D = U.subfunctions
D.interpolate(H-b)

# equation system
Dt = irksome.Dt
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
#eqn -= inner(perp(grad(inner(dv, perp(ubar)))), v)*dx
#eqn += inner(both(perp(n)*inner(dv, perp(ubar))), both(Upwind*v))*dS
#eqn += inner(dv, f*perp(ubar))*dx
eqn -= div(dv)*(
    #inner(u, u)/2
    + g*D
    + g*b
)*dx
eqn -= div(dv)*(2/3)*alpha**2*div(F)*div(F)/D/D*dx
# v-u relation
eqn += (
    inner(u
          #-v
          , du)*dx
    + (2/3)*alpha**2*div(du)*div(F)/D*dx
    )
# F definition
eqn += inner(F-
             H*u
             #D*u
             , dG)*dx
# D transport
eqn += dD*(Dt(D)
           + div(F)
           )*dx

# building a timestepper
qd = 4
method = irksome.GalerkinCollocationScheme(
    order=1,
    stage_type="deriv",
    quadrature_degree = qd,
    max_quadrature_degree = qd)

MC = irksome.MeshConstant(mesh)
dT = MC.Constant(dt)
t = MC.Constant(0.)

scheme_J = irksome.GalerkinCollocationScheme(order=1)
stepper = irksome.TimeStepper(eqn, method, t, dT, U,
                              options_prefix="stepper")#,
#scheme_J=scheme_J)

nsteps = 50

file = VTKFile('sqrtD.pvd')
v, Iu, G, D = U.subfunctions
eta = Function(Q).interpolate(D+b)
file.write(v, eta)

for step in ProgressBar('Timestep').iter(range(nsteps)):
    stepper.advance()

    file.write(v, eta)

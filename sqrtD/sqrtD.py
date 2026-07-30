from firedrake import *
import irksome

nx = 10
L = 100000
mesh = PeriodicSquareMesh(nx, nx, L)
mh = MeshHierarchy(mesh, 3)
mesh = mh[-1]

V = FunctionSpace(mesh, "BDM", 2)
Q = FunctionSpace(mesh, "DG", 1)

# v, Iu, G, D, Iu is the time integral of u
W = V * V * V * Q

# model parameters
L = Constant(L)
f = Constant(1e-4)
g = Constant(10)
H = Constant(L/20)
alpha = Constant(L/9)
dt = Constant(0.5*L/nx/sqrt(g*H))

n = FacetNormal(mesh)

# topography
b = Function(Q).interpolate(0.)

U = Function(W)

# initial conditions
_, _, _, D = U.subfunctions
x, y = SpatialCoordinate(mesh)
D.interpolate(H-b + 0.01*exp(-((x-L/2)**2 + (y-L/2)**2)/2/(L/9)**2))

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
eqn -= inner(perp(grad(inner(dv, perp(ubar)))), v)*dx
eqn += inner(both(perp(n)*inner(dv, perp(ubar))), both(Upwind*v))*dS
eqn += inner(dv, f*perp(ubar))*dx
eqn -= div(dv)*(
    inner(u, u)/2
    + g*D + g*(D+b))*dx
eqn -= div(dv)*(2/3)*alpha**2*div(F)*div(F)/D/D*dx
# v-u relation
eqn += (
    inner(u-v, du)*dx
    + (2/3)*alpha**2*div(du)*div(F)/D*dx
    )
# F definition
eqn += inner(F-D*u, dG)*dx
# D transport
eqn += dD*(Dt(D)
           + div(F)
           )*dx

# building a timestepper
qd = 10
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
                              options_prefix="stepper", scheme_J=scheme_J)

# coupled solver to construct u again for energy diagnostic
VV = V * V
du, dF = TestFunctions(VV)
uF = Function(VV)
u, F = split(uF)
uFeqn = (
    inner(u-v, du)*dx
    + (2/3)*alpha**2*div(du)*div(F)/D*dx
)
uFeqn += (
 inner(F-D*u, dF)*dx   
)
uFproblem = NonlinearVariationalProblem(uFeqn, uF)
uFsolver = NonlinearVariationalSolver(uFproblem,
                                      options_prefix="stepper")

nsteps = 200
tdump = 10
dumpt = 0

file = VTKFile('sqrtD.pvd')
v, Iu, G, D = U.subfunctions
eta = Function(Q).interpolate(D+b)
file.write(v, eta)

energy = []

for step in ProgressBar('Timestep').iter(range(nsteps)):
    F = stepper.solver._problem.F
    with assemble(F).dat.vec_ro as vec:
        res0 = vec.norm()
        snes_rtol = stepper.solver.snes.rtol
        stepper.solver.snes.ksp.atol = 0.1*snes_rtol*res0

    stepper.advance()

    dumpt += 1
    if dumpt == tdump:
        dumpt = 0
        v, _, _, D = split(U)
        uFsolver.solve()
        u, F = split(uF)
        energy0 = assemble((0.5*D*inner(u,u)+g*D*(D/2+b)+
                            alpha**2/3*inner(div(F),div(F)/D))*dx)
        energy.append(energy0)
        v, Iu, G, D = U.subfunctions
        eta.interpolate(D+b)
        file.write(v, eta)

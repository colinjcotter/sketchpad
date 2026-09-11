from firedrake import *
import irksome

nx = 10
ny = 1
Lx = 200
Ly = 10
dps = {"partition": True, "overlap_type":
       (DistributedMeshOverlapType.VERTEX, 2)}
mesh = RectangleMesh(nx, ny, Lx, Ly, originX=-Lx/2,
                     distribution_parameters=dps)
nrefs = 3
mh = MeshHierarchy(mesh, nrefs)
mesh = mh[-1]

V = FunctionSpace(mesh, "BDM", 2)
Q = FunctionSpace(mesh, "DG", 1)

# v, Iu, G, D, Iu is the time integral of u
W = V * V * V * Q

# model parameters
L = Constant(Lx)
f = Constant(0.)
g = Constant(1)
H = Constant(10)
alpha = Constant(1)
c = 5
# C_Tr = c*dt/dx so dt = C_tr*dx/c
dt = Constant(0.2*L/nx/nrefs/c)

n = FacetNormal(mesh)

# topography
b = Function(Q).interpolate(0.)

U = Function(W)

# initial conditions
v, _, _, D = U.subfunctions
x, y = SpatialCoordinate(mesh)

def sech(s):
    return 2/(exp(s) + exp(-s)) 

t0 = 0

D.interpolate(H*(1 + (c**2/(g*H) - 1))*
              sech(sqrt(3*(c**2-g*H))*(x - c*t0)/(2*c*H))**2)

u = Function(V).interpolate(as_vector([c*(1-H/D),0]))
F = Function(V).project(u*D)

du = TestFunction(V)
v0 = Function(V)
veqn = (inner(u-v0, du) + (2/3)*alpha**2*div(du)*div(F)/D)*dx
solve(veqn == 0, v0)
v.assign(v0)

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

# v, Iu, G, D, Iu is the time integral of u
ZeroV = Constant(as_vector([0,0]))
bcs = [
    DirichletBC(W.sub(0), ZeroV, "on_boundary"),
    DirichletBC(W.sub(1), ZeroV, "on_boundary"),
    DirichletBC(W.sub(2), ZeroV, "on_boundary"),
]

stepper = irksome.TimeStepper(eqn, method, t, dT, U,
                              options_prefix="stepper",
                              bcs=bcs)

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
tdump = 1
dumpt = 0

file = VTKFile('sqrtD.pvd')
_, _, _, D = split(U)
eta = Function(Q, name="Elevation").interpolate(D+b)
v, _, _, D = U.subfunctions
file.write(v, D, eta)

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
        file.write(v, D, eta)

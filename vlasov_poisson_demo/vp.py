from firedrake import *

ncells = 20
L = 2*pi
H = 6.0
nlayers = 10

# base mesh in x direction
base_mesh = PeriodicIntervalMesh(ncells, L)
# extruded mesh in x-v coordinates
mesh = ExtrudedMesh(base_mesh, layers=nlayers,
                    layer_height=H/nlayers)

# Space for the number density
V = FunctionSpace(mesh, 'DG', 1)

# Space for the electric field (independent of v)
Vbar = FunctionSpace(mesh, 'CG', 1, vfamily='R', vdegree=0)

x, v = SpatialCoordinate(mesh)

# initial condition
A = Constant(0.05)
k = Constant(0.5)
fn = Function(V).interpolate(
    (v-H/2)**2*exp(-(v-H/2)**2/2)
    #*(1 + A*cos(k*x))
)

# remove the mean
One = Function(V).assign(1.0)
fbar = assemble(fn*dx)/assemble(One*dx)
fn -= fbar

# electrostatic potential
phi = Function(Vbar)

# input for electrostatic solver
f_in = Function(V)
# Solver for electrostatic potential
psi = TestFunction(Vbar)
dphi = TrialFunction(Vbar)
phi_eqn = dphi.dx(0)*psi.dx(0)*dx - H*f_in*psi*dx
shift_eqn = dphi.dx(0)*psi.dx(0)*dx + dphi*psi*dx
nullspace = VectorSpaceBasis(constant=True)
phi_problem = LinearVariationalProblem(lhs(phi_eqn), rhs(phi_eqn),
                                       phi, aP=shift_eqn)
params = {
    'ksp_type': 'gmres',
    'pc_type': 'lu',
    'ksp_rtol': 1.0e-8,
    'ksp_monitor': None,
}
phi_solver = LinearVariationalSolver(phi_problem,
                                     nullspace=nullspace,
                                     solver_parameters=params)

dtc = Constant(0)

# Solver for DG advection
df_out = Function(V)
q = TestFunction(V)
u = as_vector([v, -phi.dx(0)])
n = FacetNormal(mesh)
un = 0.5*(dot(u, n) + abs(dot(u, n)))
df = TrialFunction(V)
df_eqn = q*df*dx
dS = dS_h + dS_v
df_eqn += dtc*((inner(u, grad(q))*f_in)*dx
               - (q('+') - q('-'))*(un('+')*f_in('+') - un('-')*f_in('-'))*dS)
df_problem = LinearVariationalProblem(lhs(df_eqn), rhs(df_eqn), df_out)
df_solver = LinearVariationalSolver(df_problem)

T = 10.0 # maximum timestep
t = 0. # model time
ndump = 10
dumpn = 0
nsteps = 10000
dt = T/nsteps
dtc.assign(dt)

# RK stage variables
f1 = Function(V)
f2 = Function(V)

outfile = VTKFile("vlasov.pvd")
f_in.assign(fn)
phi_solver.solve()
outfile.write(fn, phi)
phi.assign(.0)

for step in ProgressBar("Timestep").iter(range(nsteps)):
    f_in.assign(fn)
    phi_solver.solve()
    df_solver.solve()
    f1.assign(fn + df_out)

    f_in.assign(f1)
    phi_solver.solve()
    df_solver.solve()
    f2.assign(3*fn/4 + (f1 + df_out)/4)

    f_in.assign(f2)
    phi_solver.solve()
    df_solver.solve()
    fn.assign(fn/3 + 2*(f2 + df_out)/3)

    t += dt
    dumpn += 1
    if dumpn % ndump == 0:
        dumpn = 0
        outfile.write(fn, phi)

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

# electrostatic potential
phi = Function(Vbar)

# input for electrostatic solver
f_in = Function(V)
# Solver for electrostatic potential
psi = TestFunction(Vbar)
phi_eqn = phi.dx(0)*psi.dx(0)*dx - f_in*psi*dx
dphi = TrialFunction(Vbar)
shift_eqn = dphi.dx(0)*psi.dx(0)*dx + dphi*psi*dx
nullspace = VectorSpaceBasis(constant=True)
phi_problem = NonlinearVariationalProblem(phi_eqn, phi,
                                          Jp=shift_eqn
                                          )
params = {
    'snes_type': 'ksponly',
    'ksp_type': 'gmres',
    'pc_type': 'lu',
    'ksp_rtol': 1.0e-8,
}
phi_solver = NonlinearVariationalSolver(phi_problem,
                                        nullspace=nullspace,
                                        solver_parameters=params)

dtc = Constant(0)

# Solver for DG advection
df_out = Function(V)
q = TestFunction(V)
u = as_vector([v, -phi.dx(0)])
n = FacetNormal(mesh)
un = 0.5*(dot(u, n) + abs(dot(u, n)))
df_eqn = q*df_out*dx
dS = dS_h + dS_v
df_eqn += dtc*((inner(u, grad(q))*f_in)*dx
               - (q('+') - q('-'))*(un('+')*f_in('+') - un('-')*f_in('-'))*dS)
df_problem = NonlinearVariationalProblem(df_eqn, df_out)
df_solver = NonlinearVariationalSolver(df_problem)

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

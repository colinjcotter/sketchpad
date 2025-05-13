from firedrake import *

ncells = 50
L = 4*pi
H = 10.0
nlayers = 50

# base mesh in x direction
base_mesh = PeriodicIntervalMesh(ncells, L)
# extruded mesh in x-v coordinates
mesh = ExtrudedMesh(base_mesh, layers=nlayers,
                    layer_height=H/nlayers)

# move the mesh in the vertical so v=0 is in the middle
Vc = mesh.coordinates.function_space()
x, v = SpatialCoordinate(mesh)
X = Function(Vc).interpolate(as_vector([x, v-H/2]))
mesh.coordinates.assign(X)

# Space for the number density
V = FunctionSpace(mesh, 'DG', 1)

# Space for the electric field (independent of v)
Vbar = FunctionSpace(mesh, 'CG', 1, vfamily='R', vdegree=0)

x, v = SpatialCoordinate(mesh)

# initial condition
A = Constant(0.05)
k = Constant(0.5)
fn = Function(V).interpolate(
    v**2*exp(-v**2/2)
    *(1 + A*cos(k*x))/(2*pi)**0.5
)

# remove the mean
One = Function(V).assign(1.0)
fbar = assemble(fn*dx)/assemble(One*dx)

# electrostatic potential
phi = Function(Vbar)

# input for electrostatic solver
f_in = Function(V)
# Solver for electrostatic potential
psi = TestFunction(Vbar)
dphi = TrialFunction(Vbar)
phi_eqn = dphi.dx(0)*psi.dx(0)*dx - H*(f_in-fbar)*psi*dx
shift_eqn = dphi.dx(0)*psi.dx(0)*dx + dphi*psi*dx
nullspace = VectorSpaceBasis(constant=True)
phi_problem = LinearVariationalProblem(lhs(phi_eqn), rhs(phi_eqn),
                                       phi, aP=shift_eqn)
params = {
    'ksp_type': 'gmres',
    'pc_type': 'lu',
    'ksp_rtol': 1.0e-8,
    #'ksp_monitor': None,
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
df_a = q*df*dx
dS = dS_h + dS_v
f_bc = Function(V).assign(0.)
df_L = dtc*(div(u*q)*f_in*dx
             - (q('+') - q('-'))*(un('+')*f_in('+') - un('-')*f_in('-'))*dS
            - conditional(dot(u, n) > 0, q*dot(u, n)*f_in, 0.)*ds_tb
            )
df_problem = LinearVariationalProblem(df_a, df_L, df_out)
df_solver = LinearVariationalSolver(df_problem)

T = 50.0 # maximum timestep
t = 0. # model time
ndump = 100
dumpn = 0
nsteps = 5000
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

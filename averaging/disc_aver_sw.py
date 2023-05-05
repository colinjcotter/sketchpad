#get command arguments
import argparse
parser = argparse.ArgumentParser(description='Williamson 5 testcase for averaged propagator.')
parser.add_argument('--ref_level', type=int, default=3, help='Refinement level of icosahedral grid. Default 3.')
parser.add_argument('--tmax', type=float, default=360, help='Final time in hours. Default 24x15=360.')
parser.add_argument('--dumpt', type=float, default=6, help='Dump time in hours. Default 6.')
parser.add_argument('--dt', type=float, default=3, help='Timestep in hours. Default 3.')
parser.add_argument('--ns', type=int, default=10, help='Number of s steps in exponential approximation')
parser.add_argument('--alpha', type=float, default=1, help='Averaging window width as a multiple of dt. Default 1.')
parser.add_argument('--filename', type=str, default='w2')

args = parser.parse_known_args()
args = args[0]

hours = args.dt
dt = 60*60*hours
alpha = args.alpha #averaging window is [-alpha*dt, alpha*dt]
dt_s = Constant(alpha*dt/args.ns)
dT = Constant(dt)

from firedrake import *
import numpy as np

from firedrake.petsc import PETSc
print = PETSc.Sys.Print

#some domain, parameters and FS setup
R0 = 6371220.
H = Constant(5960.)

mesh = IcosahedralSphereMesh(radius=R0,
                             refinement_level=ref_level, degree=3,
                             comm = ensemble.comm)
cx = SpatialCoordinate(mesh)
mesh.init_cell_orientations(cx)

cx, cy, cz = SpatialCoordinate(mesh)

outward_normals = CellNormal(mesh)
perp = lambda u: cross(outward_normals, u)
    
V1 = FunctionSpace(mesh, "BDM", 2)
V2 = FunctionSpace(mesh, "DG", 1)
W = MixedFunctionSpace((V1, V2))

u, eta = TrialFunctions(W)
v, phi = TestFunctions(W)

Omega = Constant(7.292e-5)  # rotation rate
f = 2*Omega*cz/Constant(R0)  # Coriolis parameter
g = Constant(9.8)  # Gravitational constant
b = Function(V2, name="Topography")
c = sqrt(g*H)

#Set up the forward and backwards scatter discrete exponential operator
W0 = Function(W)
W1 = Function(W)
u0, eta0 = W0.subfunctions
u1, eta1 = W1.subfunctions
#D = eta + b

uh = (u0+u1)/2
etah = (eta0+eta1)/2

v, phi = TestFunctions(W)

F = (
    inner(v, u1 - u0) + dt_s*inner(f*perp(uh),v) - dt_s*g*etah*div(v)
    + phi*(eta1 - eta0) + dt_s*H*div(uh)*phi
)*dx

params = {
    'ksp_type': 'preonly',
    'mat_type': 'aij',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps'
}

# Set up the forward scatter
forward_expProb = NonlinearVariationalProblem(F, w1)
forward_expsolver = NonlinearVariationalSolver(forward_expProb,
                                               solver_parameters=params)
# Set up the backward scatter
backward_expProb = NonlinearVariationalProblem(F, w0)
backward_expsolver = NonlinearVariationalSolver(backward_expProb,
                                                solver_parameters=params)

# Set up the nonlinear operator W -> N(W)
gradperp = lambda f: perp(grad(f))
n = FacetNormal(mesh)
Upwind = 0.5 * (sign(dot(u0, n)) + 1)
both = lambda u: 2*avg(u)
K = 0.5*inner(u0, u0)
uup = 0.5 * (dot(u0, n) + abs(dot(u0, n)))

N = Function(W)
nu, neta = N.subfunctions

vector_invariant = True
L = inner(nu, v)*dx + neta*phi*dx
if vector_invariant:
    L -= (
        + inner(perp(grad(inner(v, perp(u0)))), u0)*dx
        - inner(both(perp(n)*inner(v, perp(u0))),
                both(Upwind*u0))*dS
        + div(v)*K*dx
        + inner(grad(phi), u0*(eta0-b))*dx
        - jump(phi)*(uup('+')*(eta0('+')-b('+'))
                     - uup('-')*(eta0('-') - b('-')))*dS
    )
else:
    L -= (
        + inner(div(outer(u0, v)), u0)*dx
        - inner(both(inner(n, u0)*v), both(Upwind*u0))*dS
        + inner(grad(phi), u0*(eta0-b))*dx
        - jump(phi)*(uup('+')*(eta0('+')-b('+'))
                     - uup('-')*(eta0('-') - b('-')))*dS
    )


#with topography, D = H + eta - b

NProb = NonlinearVariationalProblem(L, N)
NSolver = LinearVariationalSolver(NProb,
                                  solver_parameters = params)

# Set up the backward gather
X0 = Function(W)
X1 = Function(W)
u0, eta0 = X0.subfunctions
u1, eta1 = X1.subfunctions

F = (
    inner(v, u1 - u0) + dt_s*inner(f*perp(uh),v) - dt_s*g*etah*div(v)
    + phi*(eta1 - eta0) + dt_s*H*div(uh)*phi
)*dx
F += 



t = 0.
tmax = 60.*60.*args.tmax
dumpt = args.dumpt*60.*60.
tdump = 0.

svals = np.arange(0.5, Mbar)/Mbar #tvals goes from -rho*dt/2 to rho*dt/2
weights = np.exp(-1.0/svals/(1.0-svals))
weights = weights/np.sum(weights)
svals -= 0.5

rank = ensemble.ensemble_comm.rank
expt = rho*dt*svals[rank]
wt = weights[rank]
print(wt,"weight",expt)

x = SpatialCoordinate(mesh)

u_0 = 20.0  # maximum amplitude of the zonal wind [m/s]
u_max = Constant(u_0)
u_expr = as_vector([-u_max*x[1]/R0, u_max*x[0]/R0, 0.0])
eta_expr = - ((R0 * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R0*R0)))/g
un = Function(V1, name="Velocity").project(u_expr)
etan = Function(V2, name="Elevation").project(eta_expr)

# Topography
rl = pi/9.0
lambda_x = atan_2(x[1]/R0, x[0]/R0)
lambda_c = -pi/2.0
phi_x = asin(x[2]/R0)
phi_c = pi/6.0
minarg = Min(pow(rl, 2), pow(phi_x - phi_c, 2) + pow(lambda_x - lambda_c, 2))
bexpr = 2000.0*(1 - sqrt(minarg)/rl)
b.interpolate(bexpr)

un1 = Function(V1)
etan1 = Function(V1)

U = Function(W)
eU = Function(W)
DU = Function(W)
V = Function(W)

U_u, U_eta = U.split()
U_u.assign(un)
U_eta.assign(etan)

name = args.filename
if rank==0:
    file_sw = File(name+'.pvd', comm=ensemble.comm)
    file_sw.write(un, etan, b)

nonlinear = args.nonlinear

print ('tmax', tmax, 'dt', dt)
while t < tmax + 0.5*dt:
    print(t)
    t += dt
    tdump += dt

    if nonlinear:

        #first order splitting
        # U_{n+1} = \Phi(\exp(tL)U_n)
        #         = \exp(tL)(U_n + \exp(-tL)\Delta\Phi(\exp(tL)U_n))
        #averaged version
        # U_{n+1} = \exp(tL)(U_n + \int \rho\exp(-sL)\Delta\Phi(\exp(sL)U_n))ds

        #apply forward transformation and put result in V, storing copy in eU
        cheby.apply(U, eU, expt)
        V.assign(eU)
        
        #apply forward slow step to V
        #using sub-cycled SSPRK2

        for i in range(ncycles):
            USlow_in.assign(V)
            SlowSolver.solve()
            USlow_in.assign(USlow_out)
            SlowSolver.solve()
            V.assign(0.5*(V + USlow_out))
        #compute difference from initial value
        V -= eU

        #apply backwards transformation, put result in DU
        #without filtering
        cheby.apply(V, DU, -expt)
        DU *= wt

        #average into V
        ensemble.allreduce(DU, V)
        U += V

    V.assign(U)

    #transform forwards to next timestep
    cheby2.apply(V, U, dt)

    if rank == 0:
        if tdump > dumpt - dt*0.5:
            un.assign(U_u)
            etan.assign(U_eta)
            file_sw.write(un, etan, b)
            tdump -= dumpt

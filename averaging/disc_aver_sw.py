from firedrake import *
import numpy as np

from firedrake.petsc import PETSc
print = PETSc.Sys.Print

#get command arguments
import argparse
parser = argparse.ArgumentParser(description='Williamson 5 testcase for averaged propagator.')
parser.add_argument('--ref_level', type=int, default=3, help='Refinement level of icosahedral grid. Default 3.')
parser.add_argument('--tmax', type=float, default=360, help='Final time in hours. Default 24x15=360.')
parser.add_argument('--dumpt', type=float, default=6, help='Dump time in hours. Default 6.')
parser.add_argument('--dt', type=float, default=3, help='Timestep in hours. Default 3.')
parser.add_argument('--ns', type=int, default=10, help='Number of s steps in exponential approximation for average')
parser.add_argument('--nt', type=int, default=10, help='Number of t steps in exponential approximation for time propagator')
parser.add_argument('--alpha', type=float, default=1, help='Averaging window width as a multiple of dt. Default 1.')
parser.add_argument('--filename', type=str, default='w2')

args = parser.parse_known_args()
args = args[0]

ref_level = args.ref_level
hours = args.dt
dt = 60*60*hours
alpha = args.alpha #averaging window is [-alpha*dt, alpha*dt]
dts = alpha*dt/args.ns
dt_s = Constant(dts)
ns = args.ns
nt = args.nt

#some domain, parameters and FS setup
R0 = 6371220.
H = Constant(5960.)

mesh = IcosahedralSphereMesh(radius=R0,
                             refinement_level=ref_level, degree=3)
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
u0, eta0 = split(W0)
u1, eta1 = split(W1)
#D = eta + b

uh = (u0+u1)/2
etah = (eta0+eta1)/2

v, phi = TestFunctions(W)
Fsign = Constant(1.0)

dt_ss = dt_s*Fsign
F = (
    inner(v, u1 - u0) + dt_ss*inner(f*perp(uh),v) - dt_ss*g*etah*div(v)
    + phi*(eta1 - eta0) + dt_ss*H*div(uh)*phi
)*dx

params = {
    'ksp_type': 'preonly',
    'mat_type': 'aij',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps'
}

# Set up the forward scatter
forward_expProb = NonlinearVariationalProblem(F, W1)
forward_expsolver = NonlinearVariationalSolver(forward_expProb,
                                               solver_parameters=params)
# Set up the backward scatter
backward_expProb = NonlinearVariationalProblem(F, W0)
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
nu, neta = split(N)

vector_invariant = True
L = inner(nu, v)*dx + neta*phi*dx
if vector_invariant:
    L -= (
        + inner(perp(grad(inner(v, perp(u1)))), u1)*dx
        - inner(both(perp(n)*inner(v, perp(u1))),
                both(Upwind*u1))*dS
        + div(v)*K*dx
        + inner(grad(phi), u1*(eta1-b))*dx
        - jump(phi)*(uup('+')*(eta1('+')-b('+'))
                     - uup('-')*(eta1('-') - b('-')))*dS
    )
else:
    L -= (
        + inner(div(outer(u1, v)), u1)*dx
        - inner(both(inner(n, u1)*v), both(Upwind*u1))*dS
        + inner(grad(phi), u1*(eta1-b))*dx
        - jump(phi)*(uup('+')*(eta1('+')-b('+'))
                     - uup('-')*(eta1('-') - b('-')))*dS
    )


#with topography, D = H + eta - b

NProb = NonlinearVariationalProblem(L, N)
NSolver = NonlinearVariationalSolver(NProb,
                                  solver_parameters = params)

# Set up the backward gather
X0 = Function(W)
X1 = Function(W)
u0, eta0 = split(X0)
u1, eta1 = split(X1)

# exph(L ds) = (1 - ds/2*L)^{-1}(1 + ds/2*L)
# exph(-L ds) = (1 + ds/2*L)^{-1}(1 - ds/2*L)

# X^k = sum_{m=k}^M w_m (exph(-L ds))^m N(W_m)
# X^{M+1} = 0
# X^{k-1} = exph(-L ds)X^k + w_{k-1}*N(W_{k-1})
# (1 + ds/2*L)X^{k-1} = (1 - ds/2*L)X^k + w_k*(1 + ds/2*L)N(W_{k-1})
# X^k - X^{k-1] - ds*L(X^{k-1/2} + w_k N(W_{k-1})/2) + w_k W_{k-1} = 0

# we propagate W back, compute N, use to propagate X back

w_k = Constant(1.0) # the weight
uh = (u0 + u1 + w_k*nu)/2
etah = (eta0 + eta1 + w_k*neta)/2

F = (
    inner(v, u1 - u0) + dt_ss*inner(f*perp(uh),v) - dt_ss*g*etah*div(v)
    + phi*(eta1 - eta0) + dt_ss*H*div(uh)*phi
)*dx
F += (inner(v, w_k*nu) + phi*w_k*neta)*dx

XProb = NonlinearVariationalProblem(F, X0)
Xsolver = NonlinearVariationalSolver(XProb,
                                  solver_parameters = params)

# total number of points is ns + 1, because we have s=0
# after forward loop, W1 contains value at time ns*ds
# if we start with X^{ns+1}=0, then according to above
# (1 + ds/2*L)X^{ns} = w_k(1+ds/2*L)N(W_{ns})
# which is equivalent to X^{ns} = N(W_{ns})
# so everything is working
# we just need to change the order to
# compute N, use to propagate X back, propagate W back
# don't need to propagate W back on last step though

svals = (0.5 + np.arange(ns+1)/ns)/2 #tvals goes from -rho*dt/2 to rho*dt/2
weights = np.exp(-1.0/svals/(1.0-svals))
weights[0] /= 2
weights[-1] = 0.
weights = weights/np.sum(weights)/2

# Function to take in current state V and return dV/dt
def average(V, dVdt, forward=True):
    if forward:
        Fsign.assign(1.0)
    else:
        Fsign.assign(-1.0)
    W0.assign(V)
    # forward scatter
    dt_s.assign(dts)
    for step in range(ns):
        forward_expsolver.solve()
        W0.assign(W1)
    # backwards gather
    X1.assign(0.)
    for step in range(ns, -1, -1):
        # compute N
        NSolver.solve()
        # propagate X back
        Xsolver.solve()
        X1.assign(X0)
        # back propagate W
        if step > 0:
            w_k.assign(weights[step])
            backward_expsolver.solve()
            W1.assign(W0)
    # copy contents
    dVdt.assign(X0)

# Function to apply forward propagation in t
def propagate(V_in, V_out):
    W0.assign(V_in)
    # forward scatter
    dt_s.assign(dt)
    for step in range(nt):
        forward_expsolver.solve()
        W0.assign(W1)
    # copy contents
    V_out.assign(W1)

t = 0.
tmax = 60.*60.*args.tmax
dumpt = args.dumpt*60.*60.
tdump = 0.

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

U0 = Function(W)
U1 = Function(W)
Ustar = Function(W)
Average = Function(W)

# set up initial conditions
U_u, U_eta = U0.split()
U_u.assign(un)
U_eta.assign(etan)

name = args.filename
file_sw = File(name+'.pvd')
file_sw.write(un, etan, b)

print ('tmax', tmax, 'dt', dt)
while t < tmax + 0.5*dt:
    print(t)
    t += dt
    tdump += dt

    # V_t = <exp(-(t+s)L)N(exp((t+s)L)V)>_s
    
    # V^* = V^n + dt*f(V^n)
    # V^{n+1} = V^n + dt*f((V^n + V^*)/2)

    # V = exp(-tL)U
    # U = exp(tL)V

    # STEP 1
    # t is time at start of timestep
    # exp(-(t+dt)L)U^* = exp(-tL)U^n + dt*<exp(-(t+s)L)N(
    #                                         exp((t+s)L)V^n)>_s
    # exp(-(t+dt)L)U^* = exp(-tL)U^n + dt*<exp(-(t+s)L)N(exp(sL)U^n)>_s
    # so
    # U^* = exp(dt L)[ U^n + dt*<exp(-sL)N(exp(sL)U^n)>_s]

    # STEP 2
    # exp(-(t+dt)L)U^{n+1} = exp(-tL)U^n + dt*<
    #                  exp(-(t+s)L)N(exp((t+s)L)V^n)/2
    #                +exp(-(t+dt+s)L)N(exp((t+dt+s)L)V^*)/2>
    # so
    # exp(-(t+dt)L)U^{n+1} = exp(-tL)U^n + dt*<
    #                  exp(-(t+s)L)N(exp(sL)U^n)/2
    #                +exp(-(t+dt+s)L)N(exp(sL)U^*)/2>
    # s0
    # U^{n+1} = exp(dt L)[U^n + dt*<
    #                  exp(-sL)N(exp(sL)U^n)>/2]
    #                +dt*<exp(-sL)N(exp(sL)U^*)>/2
    #         = exp(dt L)U^n/2 + U^*/2 + dt*<exp(-sL)N(exp(sL)U^*)>/2

    # steps are then
    # Compute B^n = exp(dt L)U^n
    propagate(U0, U1)
    U1 /= 2
    # Compute U^* = exp(dt L)[ U^n + dt*<exp(-sL)N(exp(sL)U^n)>_s]
    average(U0, Average, forward=True)
    Ustar.assign(U0 + dt*Average)
    average(U0, Average, forward=False)
    Ustar += dt*Average
    # compute U^{n+1} = (B^n + U^*)/2 + dt*<exp(-sL)N(exp(sL)U^*)>/2
    average(Ustar, Average, forward=True)
    U1 += Ustar/2 + dt*Average/2
    average(Ustar, Average, forward=False)
    U1 += dt*Average/2
    # start all over again
    U0.assign(U1)

    if tdump > dumpt - dt*0.5:
        un.assign(U_u)
        etan.assign(U_eta)
        file_sw.write(un, etan, b)
        tdump -= dumpt

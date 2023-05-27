from firedrake import *
import numpy as np

from firedrake.petsc import PETSc
print = PETSc.Sys.Print

#get command arguments
import argparse
parser = argparse.ArgumentParser(description='Testing the implicit iteration for averaging.')
parser.add_argument('--ref_level', type=int, default=3, help='Refinement level of icosahedral grid. Default 3.')
#parser.add_argument('--tmax', type=float, default=360, help='Final time in hours. Default 24x15=360.')
parser.add_argument('--dumpt', type=float, default=6, help='Dump time in hours. Default 6.')
parser.add_argument('--dt', type=float, default=3, help='Timestep in hours. Default 3.')
parser.add_argument('--ns', type=int, default=10, help='Number of s steps in exponential approximation for average')
parser.add_argument('--nt', type=int, default=10, help='Number of t steps in exponential approximation for time propagator')
parser.add_argument('--alpha', type=float, default=1, help='Averaging window width as a multiple of dt. Default 1.')
parser.add_argument('--filename', type=str, default='w2', help='filename for pvd')
parser.add_argument('--check', action="store_true", help='print out some information about frequency resolution and exit')

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

if args.check:
    eigs = [0.003465, 0.007274, 0.014955] #maximum frequency for ref 3-5
    eig = eigs[ref_level-3]
    # if we are doing exp(tL) then minimum period is period of exp(it*eig)
    # which is 2*pi/eig
    min_period = 2*np.pi/eig
    print("ds steps per min wavelength", min_period/dts)
    print("dt steps per min wavelength", min_period/dt*nt)
    import sys; sys.exit()

#some domain, parameters and FS setup
R0 = 6371220.
H = Constant(5960.)

mesh = IcosahedralSphereMesh(radius=R0,
                             refinement_level=ref_level, degree=1)
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

v, phi = TestFunctions(W)
u, eta = TrialFunctions(W)

#Set up the forward and backwards scatter discrete exponential operator
W0 = Function(W)
W1 = Function(W)
u0, eta0 = split(W0)
u1, eta1 = split(W1)

uh = (u0+u)/2
etah = (eta0+eta)/2

dt_ss = dt_s
F1p = (
    inner(v, u - u0) + dt_ss*inner(f*perp(uh),v) - dt_ss*g*etah*div(v)
    + phi*(eta - eta0) + dt_ss*H*div(uh)*phi
)*dx

uh = (u+u1)/2
etah = (eta+eta1)/2
F0p = (
    inner(v, u1 - u) + dt_ss*inner(f*perp(uh),v) - dt_ss*g*etah*div(v)
    + phi*(eta1 - eta) + dt_ss*H*div(uh)*phi
)*dx

dt_ss = -dt_s
F1m = (
    inner(v, u - u0) + dt_ss*inner(f*perp(uh),v) - dt_ss*g*etah*div(v)
    + phi*(eta - eta0) + dt_ss*H*div(uh)*phi
)*dx

uh = (u+u1)/2
etah = (eta+eta1)/2
F0m = (
    inner(v, u1 - u) + dt_ss*inner(f*perp(uh),v) - dt_ss*g*etah*div(v)
    + phi*(eta1 - eta) + dt_ss*H*div(uh)*phi
)*dx

hparams = {
    'mat_type': 'matfree',
    'ksp_type': 'preonly',
    'pc_type': 'python',
    'pc_python_type': 'firedrake.HybridizationPC',
    'hybridization': {'ksp_type': 'preonly',
                      'pc_type': 'lu',
                      'pc_factor_mat_solver_type': 'mumps'}}
mparams = {
    #'ksp_monitor': None,
    'ksp_type': 'preonly',
    'pc_type': 'fieldsplit',
    'fieldsplit_0_ksp_type':'preonly',
    'fieldsplit_0_pc_type':'lu',
    'fieldsplit_0_pc_factor_mat_solver_type': 'mumps',
    'fieldsplit_1_ksp_type':'preonly',
    'fieldsplit_1_pc_type':'bjacobi',
    'fieldsplit_1_sub_pc_type':'ilu'
}

# Set up the forward scatter
forwardp_expProb = LinearVariationalProblem(lhs(F1p), rhs(F1p), W1,
                                            constant_jacobian=True)
forwardp_expsolver = LinearVariationalSolver(forwardp_expProb,
                                               solver_parameters=hparams)
forwardm_expProb = LinearVariationalProblem(lhs(F1m), rhs(F1m), W1,
                                            constant_jacobian=True)
forwardm_expsolver = LinearVariationalSolver(forwardm_expProb,
                                               solver_parameters=hparams)

# Set up the backward scatter
backwardp_expProb = LinearVariationalProblem(lhs(F0p), rhs(F0p), W0,
                                             constant_jacobian=True)
backwardp_expsolver = LinearVariationalSolver(backwardp_expProb,
                                                solver_parameters=hparams)
backwardm_expProb = LinearVariationalProblem(lhs(F0m), rhs(F0m), W0,
                                             constant_jacobian=True)
backwardm_expsolver = LinearVariationalSolver(backwardm_expProb,
                                                solver_parameters=hparams)

#linearised variables
dW0 = Function(W)
dW1 = Function(W)

#propagators for linearised variables
u0, eta0 = split(dW0)
u1, eta1 = split(dW1)

uh = (u0+u)/2
etah = (eta0+eta)/2

dt_ss = dt_s
F1p = (
    inner(v, u - u0) + dt_ss*inner(f*perp(uh),v) - dt_ss*g*etah*div(v)
    + phi*(eta - eta0) + dt_ss*H*div(uh)*phi
)*dx

uh = (u+u1)/2
etah = (eta+eta1)/2
F0p = (
    inner(v, u1 - u) + dt_ss*inner(f*perp(uh),v) - dt_ss*g*etah*div(v)
    + phi*(eta1 - eta) + dt_ss*H*div(uh)*phi
)*dx

dt_ss = -dt_s
F1m = (
    inner(v, u - u0) + dt_ss*inner(f*perp(uh),v) - dt_ss*g*etah*div(v)
    + phi*(eta - eta0) + dt_ss*H*div(uh)*phi
)*dx

uh = (u+u1)/2
etah = (eta+eta1)/2
F0m = (
    inner(v, u1 - u) + dt_ss*inner(f*perp(uh),v) - dt_ss*g*etah*div(v)
    + phi*(eta1 - eta) + dt_ss*H*div(uh)*phi
)*dx

# Set up the forward scatter
dforwardp_expProb = LinearVariationalProblem(lhs(F1p), rhs(F1p), dW1,
                                            constant_jacobian=True)
dforwardp_expsolver = LinearVariationalSolver(forwardp_expProb,
                                               solver_parameters=hparams)
dforwardm_expProb = LinearVariationalProblem(lhs(F1m), rhs(F1m), dW1,
                                            constant_jacobian=True)
dforwardm_expsolver = LinearVariationalSolver(forwardm_expProb,
                                               solver_parameters=hparams)

# Set up the backward scatter
dbackwardp_expProb = LinearVariationalProblem(lhs(F0p), rhs(F0p), dW0,
                                             constant_jacobian=True)
dbackwardp_expsolver = LinearVariationalSolver(backwardp_expProb,
                                                solver_parameters=hparams)
dbackwardm_expProb = LinearVariationalProblem(lhs(F0m), rhs(F0m), dW0,
                                             constant_jacobian=True)
dbackwardm_expsolver = LinearVariationalSolver(backwardm_expProb,
                                                solver_parameters=hparams)

# Set up the nonlinear operator W -> N(W)
gradperp = lambda f: perp(grad(f))
n = FacetNormal(mesh)
Upwind = 0.5 * (sign(dot(u1, n)) + 1)
both = lambda u: 2*avg(u)
K = 0.5*inner(u1, u1)
uup = 0.5 * (dot(u1, n) + abs(dot(u1, n)))

N = Function(W)
nu, neta = TrialFunctions(W)

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

NProb = LinearVariationalProblem(lhs(L), rhs(L), N,
                                 constant_jacobian=True)
NSolver = LinearVariationalSolver(NProb,
                                  solver_parameters = mparams)

# linearised operator (still solves into N)
dL = inner(nu, v)*dx + neta*phi*dx + action(derivative(L, W1), dW1)

LNProb = LinearVariationalProblem(lhs(dL), rhs(dL), N,
                                 constant_jacobian=True)
LNSolver = LinearVariationalSolver(LNProb,
                                   solver_parameters = mparams)

# linearised backward Euler solve (still solves into N)
dL = inner(nu, v)*dx + neta*phi*dx + Constant(dt)*action(derivative(L, W1), N)
LBENProb = LinearVariationalProblem(lhs(dL), rhs(dL), N,
                                    constant_jacobian=False)
LBENSolver = LinearVariationalSolver(LBENProb,
                                   solver_parameters = mparams)

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
u, eta = TrialFunctions(W)

nu, neta = split(N)

theta = Constant(0.5)
uh = (1-theta)*u + theta*u1 + (1-theta)*w_k*nu
etah = (1-theta)*eta + theta*eta1 + (1-theta)*w_k*neta

dt_ss = dt_s
Fp = (
    inner(v, u1 - u) + dt_ss*inner(f*perp(uh),v) - dt_ss*g*etah*div(v)
    + phi*(eta1 - eta) + dt_ss*H*div(uh)*phi
)*dx
Fp += (inner(v, w_k*nu) + phi*w_k*neta)*dx

XProbp = LinearVariationalProblem(lhs(Fp), rhs(Fp), X0,
                                  constant_jacobian=True)
Xpsolver = LinearVariationalSolver(XProbp,
                                  solver_parameters = hparams)

dt_ss = -dt_s
Fm = (
    inner(v, u1 - u) + dt_ss*inner(f*perp(uh),v) - dt_ss*g*etah*div(v)
    + phi*(eta1 - eta) + dt_ss*H*div(uh)*phi
)*dx
Fm += (inner(v, w_k*nu) + phi*w_k*neta)*dx

XProbm = LinearVariationalProblem(lhs(Fm), rhs(Fm), X0,
                                  constant_jacobian=True)
Xmsolver = LinearVariationalSolver(XProbm,
                                  solver_parameters = hparams)

# total number of points is 2ns + 1, because we have s=0
# after forward loop, W1 contains value at time ns*ds
# if we start with X^{ns+1}=0, then according to above
# (1 + ds/2*L)X^{ns} = w_k(1+ds/2*L)N(W_{ns})
# which is equivalent to X^{ns} = N(W_{ns})
# so everything is working
# we just need to change the order to
# compute N, use to propagate X back, propagate W back
# don't need to propagate W back on last step though

svals = 0.5 + np.arange(2*ns)/2/ns/2 #tvals goes from -rho*dt/2 to rho*dt/2
weights = np.exp(-1.0/svals/(1.0-svals))
weights = weights[ns:]
weights[0] /= 2
weights = weights/np.sum(weights)/2
weights = np.concatenate((weights, [0]))

# Function to take in current state V and return dV/dt
def average(V, dVdt, positive=True):
    W0.assign(V)
    # forward scatter
    dt_s.assign(dts)
    for step in ProgressBar(f'average forward').iter(range(ns)):
        if positive:
            forwardp_expsolver.solve()
        else:
            forwardm_expsolver.solve()
        W0.assign(W1)
    # backwards gather
    X1.assign(0.)
    for step in ProgressBar(f'average backward').iter(range(ns, -1, -1)):
        # compute N
        NSolver.solve()
        # propagate X back
        if positive:
            Xpsolver.solve()
        else:
            Xmsolver.solve()
        X1.assign(X0)
        # back propagate W
        if step > 0:
            w_k.assign(weights[step])
            if positive:
                backwardp_expsolver.solve()
            else:
                backwardm_expsolver.solve()
            W1.assign(W0)
    # copy contents
    dVdt.assign(X0)
    
# Function to take in a perturbation dV and return averaged
# linearisation of backward Euler action_dN_dV at V

def average_linear(V, dV, action_dN_dV, positive=True, PC=False):
    dW0.assign(dV)
    W0.assign(dV)
    # forward scatter
    dt_s.assign(dts)
    for step in ProgressBar(f'average linear forward').iter(range(ns)):
        if positive:
            # forward scatter linear
            forwardp_expsolver.solve()
            dforwardp_expsolver.solve()
        else:
            forwardm_expsolver.solve()
            dforwardm_expsolver.solve()
        W0.assign(W1)
        dW0.assign(dW1)
    # backwards gather
    X1.assign(0.)
    for step in ProgressBar(f'average linear backward').iter(range(ns, -1, -1)):
        # Apply linearised operator
        if PC:
            LBENSolver.solve()
        else:
            LNSolver.solve()
            N *= Constant(dt)
            N += dW1
        # propagate X back
        if positive:
            Xpsolver.solve()
        else:
            Xmsolver.solve()
        X1.assign(X0)
        # back propagate W
        if step > 0:
            w_k.assign(weights[step])
            if positive:
                backwardp_expsolver.solve()
                dbackwardp_expsolver.solve()
            else:
                backwardm_expsolver.solve()
                dbackwardm_expsolver.solve()
            W1.assign(W0)
            dW1.assign(dW0)
    # copy contents
    action_dN_dV.assign(X0)

# Function to apply forward propagation in t
def propagate(V_in, V_out):
    W0.assign(V_in)
    # forward scatter
    dt_s.assign(dt/nt)
    for step in ProgressBar(f'propagate').iter(range(nt)):
        forwardp_expsolver.solve()
        W0.assign(W1)
    # copy contents
    V_out.assign(W1)

t = 0.
#tmax = 60.*60.*args.tmax
#dumpt = args.dumpt*60.*60.
#tdump = 0.

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
minarg = min_value(pow(rl, 2), pow(phi_x - phi_c, 2) + pow(lambda_x - lambda_c, 2))
bexpr = 2000.0*(1 - sqrt(minarg)/rl)
b.interpolate(bexpr)

U0 = Function(W)
U1 = Function(W)

# set up initial conditions
U_u, U_eta = U0.subfunctions
U_u.assign(un)
U_eta.assign(etan)

U1.assign(U0)

# Some working memory

RNon = Function(W) # nonlinear residual
RLin = Function(W) # linear residual
Average = Function(W) # a place to put the average
dUk = Function(W) # current estimate of linear increment
dUkp1 = Function(V) # next estimate of linear increment

# backward Euler notes

# V_t = <exp(-(t+s)L)N(exp((t+s)L)V)>_s
# V^{n+1} = V^n + dt*<exp(-(t+dt+s)L)N(exp((t+dt+s)L)V^{n+1})>_s

# t is time at start of timestep
# V^{n+1} = exp(-(t+dt)L)U^{n+1}
# U^{n+1} = exp((t+dt)L)V^{n+1}

# exp(-(t+dt)L)U^{n+1} = exp(-tL)U^n + dt*<exp(-(t+dt+s)L)N(exp(sL)U^{n+1})>_s
# so
# U^{n+1} = exp(-dt*L)U^n + dt*<exp(-sL)N(exp(sL)U^{n+1})>_s

# Compute exp(dt L)U^n
propagate(U0, U0)

# assemble backward Euler residual
average(U0, Average, positive=True)
RNon.assign(U0 + dt*Average)
average(U0, Average, positive=False)
RNon += dt*Average
RNon *= -1

kmax = 5
dUk.assign(0.)
for k in range(kmax):
    # solving Ax = b by iterating
    # Mx^{k+1} = (M-A)x^k + b
    # rewrite as
    # x^{k+1} = x^k + M^{-1}(b - Ax^k)

    # compute linear residual
    RLin.assign(RNon)
    average_linear(dUk, average, positive=True)
    RLin -= average
    average_linear(dUk, average, positive=False)
    RLin -= average

    # Compute residual
    COMPUTE RESIDUAL
    
    # apply the preconditioner
    average_linear(RLin, average, positive=True, PC=True)
    dUk -= average
    average_linear(RLin, average, positive=False, PC=True)
    dUk -= average

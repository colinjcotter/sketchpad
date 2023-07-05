from firedrake import *
import numpy as np
import mg

from firedrake.petsc import PETSc
print = PETSc.Sys.Print

#get command arguments
import argparse
parser = argparse.ArgumentParser(description='Williamson 5 testcase for averaged propagator using D (thickness) as the pressure variable.')
parser.add_argument('--ref_level', type=int, default=3, help='Refinement level of icosahedral grid. Default 3.')
parser.add_argument('--tmax', type=float, default=360, help='Final time in hours. Default 24x15=360.')
parser.add_argument('--dumpt', type=float, default=6, help='Dump time in hours. Default 6.')
parser.add_argument('--dt', type=float, default=3, help='Timestep in hours. Default 3.')
parser.add_argument('--ns', type=int, default=10, help='Number of s steps in exponential approximation for average')
parser.add_argument('--nt', type=int, default=10, help='Number of t steps in exponential approximation for time propagator')
parser.add_argument('--alpha', type=float, default=1, help='Averaging window width as a multiple of dt. Default 1.')
parser.add_argument('--filename', type=str, default='w2', help='filename for pvd')
parser.add_argument('--check', action="store_true", help='print out some information about frequency resolution and exit')
parser.add_argument('--advection', action="store_true", help='include mean flow advection in L.')
parser.add_argument('--dynamic_ubar', action="store_true", help='Use un as ubar.')
parser.add_argument('--vector_invariant', action="store_true", help='use vector invariant form.')
parser.add_argument('--eta', action="store_true", help='use eta instead of D.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

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
H0 = Constant(5960.)

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

One = Function(V2).assign(1.0)
Area = assemble(One*dx)

u, D = TrialFunctions(W)
v, phi = TestFunctions(W)

Omega = Constant(7.292e-5)  # rotation rate
f = 2*Omega*cz/Constant(R0)  # Coriolis parameter
g = Constant(9.8)  # Gravitational constant
b = Function(V2, name="Topography")
c = sqrt(g*H0)

if args.eta:
    H = H0 - b
else:
    H = H0

#Set up the forward and backwards scatter discrete exponential operator
W0 = Function(W)
W1 = Function(W)
u0, D0 = split(W0)
u1, D1 = split(W1)
#D = eta + b

v, phi = TestFunctions(W)

if args.advection:
    ubar = Function(V1)

if args.dynamic_ubar:
    constant_jacobian = True
else:
    constant_jacobian = False

def advection(F, ubar, v, continuity=False, vector=False, upwind=True):
    """
    Advection of F by ubar using test function v
    """

    if continuity:
        L = -inner(grad(v), outer(ubar, F))*dx
    else:
        L = -inner(div(outer(v, ubar)), F)*dx
    n = FacetNormal(mesh)
    if upwind:
        un = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))
    else:
        un = 0.5*(dot(ubar, n) - abs(dot(ubar, n)))
    L += dot(jump(v), (un('+')*F('+') - un('-')*F('-')))*dS
    if vector:
        L += un('+')*inner(v('-'), n('+')+n('-'))*inner(F('+'), n('+'))*dS
        L += un('-')*inner(v('+'), n('+')+n('-'))*inner(F('-'), n('-'))*dS
    return L

u, D = TrialFunctions(W)
uh = (u0+u)/2
Dh = (D0+D)/2

dt_ss = dt_s
# positive s outward propagation
F1p = (
    inner(v, u - u0) + dt_ss*inner(f*perp(uh),v) - dt_ss*g*Dh*div(v)
    + phi*(D - D0) + dt_ss*H*div(uh)*phi
)*dx

if args.advection:
    F1p += dt_ss*advection(uh, ubar, v, vector=True)
    F1p += dt_ss*advection(Dh, ubar, phi,
                           continuity=True, vector=False)

uh = (u+u1)/2
Dh = (D+D1)/2
# positive s inward propagation
F0p = (
    inner(v, u1 - u) + dt_ss*inner(f*perp(uh),v) - dt_ss*g*Dh*div(v)
    + phi*(D1 - D) + dt_ss*H*div(uh)*phi
)*dx

if args.advection:
    F0p += dt_ss*advection(uh, ubar, v, vector=True, upwind=False)
    F0p += dt_ss*advection(Dh, ubar, phi, vector=False,
                           continuity=True, upwind=False)

dt_ss = -dt_s
# negative s outward  propagation
F1m = (
    inner(v, u - u0) + dt_ss*inner(f*perp(uh),v) - dt_ss*g*Dh*div(v)
    + phi*(D - D0) + dt_ss*H*div(uh)*phi
)*dx

if args.advection:
    F1m += dt_ss*advection(uh, ubar, v, vector=True, upwind=False)
    F1m += dt_ss*advection(Dh, ubar, phi, vector=False,
                           continuity=True, upwind=False)

uh = (u+u1)/2
Dh = (D+D1)/2
# negative s inward propagation
F0m = (
    inner(v, u1 - u) + dt_ss*inner(f*perp(uh),v) - dt_ss*g*Dh*div(v)
    + phi*(D1 - D) + dt_ss*H*div(uh)*phi
)*dx

if args.advection:
    F0m += dt_ss*advection(uh, ubar, v, vector=True)
    F0m += dt_ss*advection(Dh, ubar, phi, continuity=True, vector=False)

hparams = {
    #"snes_view": None,
    #"snes_lag_preconditioner": 10,
    #"snes_lag_preconditioner_persists": None,
    'mat_type': 'matfree',
    'ksp_type': 'gmres',
    #'ksp_monitor': None,
    #'ksp_converged_reason': None,
    'pc_type': 'python',
    'pc_python_type': 'firedrake.HybridizationPC',
    'hybridization': {'ksp_type': 'preonly',
                      'pc_type': 'lu',
                      'pc_factor_mat_solver_type': 'mumps'
                      }}
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

monoparameters_ns = {
    #"snes_monitor": None,
    "snes_lag_preconditioner": ns,
    "snes_lag_preconditioner_persists": None,
    "mat_type": "matfree",
    "ksp_type": "gmres",
    #'ksp_monitor': None,
    #"ksp_monitor_true_residual": None,
    #"ksp_converged_reason": None,
    "ksp_atol": 1e-8,
    "ksp_rtol": 1e-8,
    "ksp_max_it": 40,
    "pc_type": "python",
    "pc_python_type": "firedrake.PatchPC",
    "patch_pc_patch_save_operators": True,
    "patch_pc_patch_partition_of_unity": True,
    "patch_pc_patch_sub_mat_type": "seqdense",
    "patch_pc_patch_construct_dim": 0,
    "patch_pc_patch_construct_type": "star",
    "patch_pc_patch_local_type": "additive",
    "patch_pc_patch_precompute_element_tensors": True,
    "patch_pc_patch_symmetrise_sweep": False,
    "patch_sub_ksp_type": "preonly",
    "patch_sub_pc_type": "lu",
    "patch_sub_pc_factor_shift_type": "nonzero",
}

monoparameters_nt = {
    #"snes_monitor": None,
    "snes_lag_preconditioner": nt,
    "snes_lag_preconditioner_persists": None,
    "mat_type": "matfree",
    "ksp_type": "gmres",
    #'ksp_monitor': None,
    #"ksp_monitor_true_residual": None,
    #"ksp_converged_reason": None,
    "ksp_atol": 1e-8,
    "ksp_rtol": 1e-8,
    "ksp_max_it": 40,
    "pc_type": "python",
    "pc_python_type": "firedrake.PatchPC",
    "patch_pc_patch_save_operators": True,
    "patch_pc_patch_partition_of_unity": True,
    "patch_pc_patch_sub_mat_type": "seqdense",
    "patch_pc_patch_construct_dim": 0,
    "patch_pc_patch_construct_type": "star",
    "patch_pc_patch_local_type": "additive",
    "patch_pc_patch_precompute_element_tensors": True,
    "patch_pc_patch_symmetrise_sweep": False,
    "patch_sub_ksp_type": "preonly",
    "patch_sub_pc_type": "lu",
    "patch_sub_pc_factor_shift_type": "nonzero",
}



if args.advection:
    params = monoparameters_ns
else:
    params = hparams

# Set up the forward scatter
forwardp_expProb = LinearVariationalProblem(lhs(F1p), rhs(F1p), W1,
                                            constant_jacobian=constant_jacobian)
forwardp_expsolver = LinearVariationalSolver(forwardp_expProb,
                                               solver_parameters=params)
forwardm_expProb = LinearVariationalProblem(lhs(F1m), rhs(F1m), W1,
                                            constant_jacobian=constant_jacobian)
forwardm_expsolver = LinearVariationalSolver(forwardm_expProb,
                                               solver_parameters=params)

# Set up the forward solver for dt propagation
if args.advection:
    params = monoparameters_nt
else:
    params = hparams

u, D = TrialFunctions(W)
uh = (u0+u)/2
Dh = (D0+D)/2

dt_ss = Constant(dt/nt)
# positive s outward propagation
F1p = (
    inner(v, u - u0) + dt_ss*inner(f*perp(uh),v) - dt_ss*g*Dh*div(v)
    + phi*(D - D0) + dt_ss*H*div(uh)*phi
)*dx

if args.advection:
    F1p += dt_ss*advection(uh, ubar, v, vector=True)
    F1p += dt_ss*advection(Dh, ubar, phi,
                           continuity=True, vector=False)

forwardp_expProb_dt = LinearVariationalProblem(lhs(F1p), rhs(F1p), W1,
                                            constant_jacobian=constant_jacobian)
forwardp_expsolver_dt = LinearVariationalSolver(forwardp_expProb_dt,
                                                solver_parameters=params)

# Set up the backward scatter
if args.advection:
    params = monoparameters_ns
else:
    params = hparams

backwardp_expProb = LinearVariationalProblem(lhs(F0p), rhs(F0p), W0,
                                             constant_jacobian=constant_jacobian)
backwardp_expsolver = LinearVariationalSolver(backwardp_expProb,
                                                solver_parameters=params)
backwardm_expProb = LinearVariationalProblem(lhs(F0m), rhs(F0m), W0,
                                             constant_jacobian=constant_jacobian)
backwardm_expsolver = LinearVariationalSolver(backwardm_expProb,
                                                solver_parameters=params)

# Set up the nonlinear operator W -> N(W)
gradperp = lambda f: perp(grad(f))
n = FacetNormal(mesh)
Upwind = 0.5 * (sign(dot(u1, n)) + 1)
both = lambda u: 2*avg(u)
K = 0.5*inner(u1, u1)
uup = 0.5 * (dot(u1, n) + abs(dot(u1, n)))

N = Function(W)
nu, nD = TrialFunctions(W)

vector_invariant = args.vector_invariant
# Sign confusions! We are solving for nu, nD, but
# equation is written in the form (nu, nD) - N(u1, D1) = 0.
L = inner(nu, v)*dx + nD*phi*dx
if not args.eta:
    L -= div(v)*g*b*dx

if vector_invariant:
    L -= (
        + inner(perp(grad(inner(v, perp(u1)))), u1)*dx
        - inner(both(perp(n)*inner(v, perp(u1))),
                both(Upwind*u1))*dS
        + div(v)*K*dx
        + inner(grad(phi), u1*D1)*dx
        - jump(phi)*(uup('+')*D1('+')
                     - uup('-')*D1('-'))*dS
    )
else:
    L += advection(u1, u1, v, vector=True)
    if args.eta:
        L += advection(D1, u1, phi, continuity=True, vector=False)
    else:
        L += advection(D1-H, u1, phi, continuity=True, vector=False)

# for args.eta True we have eta_t + div(u(eta+H)) = eta_t + div(uH) + div(u*eta) [linear and nonlinear]
# otherwise we have D_t + div(uD) = D_t + div(uH) + div(u(D-H))
# noting that H = H0 - b when args.eta True and H = H0 otherwise

# combining with args.advection True
# for args.eta True we have eta_t + [div(uH) + div(ubar*eta)] + div(u*eta - ubar*eta) [linear in square brackets]
# otherwise we have D_t + [div(uH) + div(ubar*D)] + div((u(D-H) - ubar*D) [linear in square brackets]
# last term disappears when div ubar = 0.

if args.advection:
    L -= advection(u1, ubar, v, vector=True)
    if args.eta:
        L -= advection(D1, ubar, phi, continuity=True, vector=False)
    else:
        L -= advection(D1, ubar, phi, continuity=True, vector=False)

#with topography, D = H + eta - b

NProb = LinearVariationalProblem(lhs(L), rhs(L), N,
                                 constant_jacobian=True)
NSolver = LinearVariationalSolver(NProb,
                                  solver_parameters = mparams)

# Set up the backward gather
X0 = Function(W)
X1 = Function(W)
u0, D0 = split(X0)
u1, D1 = split(X1)

# exph(L ds) = (1 - ds/2*L)^{-1}(1 + ds/2*L)
# exph(-L ds) = (1 + ds/2*L)^{-1}(1 - ds/2*L)

# X^k = sum_{m=k}^M w_m (exph(-L ds))^m N(W_m)
# X^{M+1} = 0
# X^{k-1} = exph(-L ds)X^k + w_{k-1}*N(W_{k-1})
# (1 + ds/2*L)X^{k-1} = (1 - ds/2*L)X^k + w_k*(1 + ds/2*L)N(W_{k-1})
# X^k - X^{k-1] - ds*L(X^{k-1/2} + w_k N(W_{k-1})/2) + w_k W_{k-1} = 0

# we propagate W back, compute N, use to propagate X back

w_k = Constant(1.0) # the weight
u, D = TrialFunctions(W)

nu, nD = split(N)

theta = Constant(0.5)
uh = (1-theta)*u + theta*u1 + (1-theta)*w_k*nu
Dh = (1-theta)*D + theta*D1 + (1-theta)*w_k*nD

dt_ss = dt_s
# positive s inward propagation
Fp = (
    inner(v, u1 - u) + dt_ss*inner(f*perp(uh),v) - dt_ss*g*Dh*div(v)
    + phi*(D1 - D) + dt_ss*H*div(uh)*phi
)*dx
Fp += (inner(v, w_k*nu) + phi*w_k*nD)*dx
if args.advection:
    # we are going backwards in time
    Fp += dt_ss*advection(uh, ubar, v, upwind=False, vector=True)
    Fp += dt_ss*advection(Dh, ubar, phi, continuity=True,
                          upwind=False, vector=False)

XProbp = LinearVariationalProblem(lhs(Fp), rhs(Fp), X0,
                                  constant_jacobian=constant_jacobian)
Xpsolver = LinearVariationalSolver(XProbp,
                                  solver_parameters = params)

dt_ss = -dt_s
# negative s inward propagation
Fm = (
    inner(v, u1 - u) + dt_ss*inner(f*perp(uh),v) - dt_ss*g*Dh*div(v)
    + phi*(D1 - D) + dt_ss*H*div(uh)*phi
)*dx
Fm += (inner(v, w_k*nu) + phi*w_k*nD)*dx
if args.advection:
    Fm += dt_ss*advection(uh, ubar, v, vector=True)
    Fm += dt_ss*advection(Dh, ubar, phi, continuity=True, vector=False)

XProbm = LinearVariationalProblem(lhs(Fm), rhs(Fm), X0,
                                  constant_jacobian=constant_jacobian)
Xmsolver = LinearVariationalSolver(XProbm,
                                  solver_parameters = params)

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
def average(V, dVdt, positive=True, t=None):
    W0.assign(V)
    # forward scatter
    dt_s.assign(dts)
    for step in ProgressBar(f'average forward').iter(range(ns)):
        with PETSc.Log.Event("forward propagation ds"):
            if positive:
                forwardp_expsolver.solve()
            else:
                forwardm_expsolver.solve()
        W0.assign(W1)
    # backwards gather
    X1.assign(0.)
    for step in ProgressBar(f'average backward').iter(range(ns, -1, -1)):
        # compute N
        with PETSc.Log.Event("nonlinearity"):
            NSolver.solve()
        # propagate X back
        with PETSc.Log.Event("backward integration"):
            if positive:
                Xpsolver.solve()
            else:
                Xmsolver.solve()
        X1.assign(X0)
        # back propagate W
        if step > 0:
            w_k.assign(weights[step])
            with PETSc.Log.Event("backward propagation ds"):
                if positive:
                    backwardp_expsolver.solve()
                else:
                    backwardm_expsolver.solve()
            W1.assign(W0)
    # copy contents
    dVdt.assign(X0)

# Function to apply forward propagation in t
def propagate(V_in, V_out, t=None):
    W0.assign(V_in)
    # forward scatter
    for step in ProgressBar(f'propagate').iter(range(nt)):
        with PETSc.Log.Event("forward propagation dt"):
            forwardp_expsolver_dt.solve()
            W0.assign(W1)
    # copy contents
    V_out.assign(W1)

t = 0.
tmax = 60.*60.*args.tmax
#tmax = dt
dumpt = args.dumpt*60.*60.
tdump = 0.

x = SpatialCoordinate(mesh)

u_0 = 20.0  # maximum amplitude of the zonal wind [m/s]
u_max = Constant(u_0)
u_expr = as_vector([-u_max*x[1]/R0, u_max*x[0]/R0, 0.0])
eta_expr = - ((R0 * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R0*R0)))/g
un = Function(V1, name="Velocity").project(u_expr)
if args.advection:
    # projection solve removes divergent parts
    u, p = TrialFunctions(W)
    w, q = TestFunctions(W)
    eqn = (
        inner(w, u) - div(w)*p
        + q*div(u+un)
        )*dx
    Uproj = Function(W)
    projection_problem = LinearVariationalProblem(lhs(eqn), rhs(eqn), Uproj)
    v_basis = VectorSpaceBasis(constant=True, comm=COMM_WORLD)
    nullspace = MixedVectorSpaceBasis(W, [W.sub(0), v_basis])
    projection_solver = LinearVariationalSolver(projection_problem, nullspace=nullspace,
                                                solver_parameters = hparams)
    projection_solver.solve()
    u, _ = Uproj.subfunctions
    ubar.assign(un + u)

# Topography
rl = pi/9.0
lambda_x = atan_2(x[1]/R0, x[0]/R0)
lambda_c = -pi/2.0
phi_x = asin(x[2]/R0)
phi_c = pi/6.0
minarg = min_value(pow(rl, 2), pow(phi_x - phi_c, 2) + pow(lambda_x - lambda_c, 2))
bexpr = 2000.0*(1 - sqrt(minarg)/rl)
b.interpolate(bexpr)
if args.eta:
    Dn = Function(V2, name="Elevation").interpolate(eta_expr)
else:
    Dn = Function(V2, name="Depth").interpolate(eta_expr + H - b)

U0 = Function(W)
U1 = Function(W)
Ustar = Function(W)
Average = Function(W)

# set up initial conditions
U_u, U_D = U0.subfunctions
U1_u, U1_D = U1.subfunctions
U_u.assign(un)
U_D.assign(Dn)

etan = Function(V2, name="Elevation")

name = args.filename
file_sw = File(name+'.pvd')
if args.eta:
    etan.assign(Dn)
else:
    etan.assign(Dn-H0+b)
file_sw.write(un, etan, b)

mass0 = assemble(U_D*dx)

print ('tmax', tmax, 'dt', dt)
while t < tmax - 0.5*dt:
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
    print("RK stage 1")
    propagate(U0, U1, t=t)
    U1 /= 2
    # Compute U^* = exp(dt L)[ U^n + dt*<exp(-sL)N(exp(sL)U^n)>_s]
    average(U0, Average, positive=True, t=t)
    Ustar.assign(U0 + dt*Average)
    average(U0, Average, positive=False, t=t)
    Ustar += dt*Average
    propagate(Ustar, Ustar, t=t)
    # compute U^{n+1} = (B^n + U^*)/2 + dt*<exp(-sL)N(exp(sL)U^*)>/2
    print("RK stage 2")
    average(Ustar, Average, positive=True, t=t)
    U1 += Ustar/2 + dt*Average/2
    average(Ustar, Average, positive=False, t=t)
    U1 += dt*Average/2
    # start all over again
    U0.assign(U1)
    if args.dynamic_ubar:
        projection_solver.solve()
        u, _ = Uproj.subfunctions
        ubar.assign(un + u)
    print("mass error", (mass0-assemble(U_D*dx))/Area)
    
    if tdump > dumpt - dt*0.5:
        un.assign(U_u)
        Dn.assign(U_D)
        if args.eta:
            etan.assign(Dn)
        else:
            etan.assign(Dn-H0+b)
        file_sw.write(un, etan, b)
        tdump -= dumpt

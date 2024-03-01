from firedrake import *
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
import asQ

import numpy as np
print = PETSc.Sys.Print

#get command arguments
import argparse
parser = argparse.ArgumentParser(description='Williamson 5 testcase for averaged propagator using D (thickness) as the pressure variable.')
parser.add_argument('--ref_level', type=int, default=3, help='Refinement level of icosahedral grid. Default 3.')
parser.add_argument('--tmax', type=float, default=1, help='Final time in hours. Default 24x15=360.')
parser.add_argument('--dumpt', type=float, default=0.5, help='Dump time in hours. Default 6.')
parser.add_argument('--checkt', type=float, default=0.5, help='Create checkpointing file every checkt hours. Default 6.')
parser.add_argument('--dt', type=float, default=0.5, help='Timestep in hours. Default 3.')
parser.add_argument('--ns', type=int, default=4, help='Number of s steps in exponential approximation for average')
parser.add_argument('--nt', type=int, default=4, help='Number of t steps in exponential approximation for time propagator')
parser.add_argument('--alpha', type=float, default=1, help='Averaging window width as a multiple of dt. Default 1.')
parser.add_argument('--filename', type=str, default='w2', help='filename for pvd')
parser.add_argument('--check', action="store_true", help='print out some information about frequency resolution and exit')
parser.add_argument('--advection', action="store_true", help='include mean flow advection in L.')
parser.add_argument('--dynamic_ubar', action="store_true", help='Use un as ubar.')
parser.add_argument('--vector_invariant', action="store_true", help='use vector invariant form.')
parser.add_argument('--eta', action="store_true", help='use eta instead of D.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')
parser.add_argument('--linear', action='store_true', help='Just solve the linearpropagator at each step (as if N=0).')
parser.add_argument('--linear_velocity', action='store_true', help='Drop the velocity advection from N.')
parser.add_argument('--linear_height', action='store_true', help='Drop the height advection from N.')
parser.add_argument('--timestepping', type=str, default='rk4', choices=['rk4', 'heuns'], help='Choose a time steeping method. Default heuns.')
parser.add_argument('--pickup', action='store_true', help='Pickup the result from the checkpoint.')
parser.add_argument('--pickup_from', type=str, default='w2')
parser.add_argument('--nslices', type=int, default=1, help='Number of time-slices per time-window.')
parser.add_argument('--alphap', type=float, default=0.0001, help='Circulant coefficient.')

args = parser.parse_known_args()
args = args[0]
timestepping = args.timestepping

print(args)

if args.show_args:
    PETSc.Sys.Print(args)


PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Setting up --- === ###')
PETSc.Sys.Print('')

# set arguments
ref_level = args.ref_level
hours = args.dt
dt = 60*60*hours
alpha = args.alpha #averaging window is [-alpha*dt, alpha*dt]
dts = alpha*dt/args.ns
dt_s = Constant(dts)
ns = args.ns
nt = args.nt

if nt % 2 == 1:
    print('nt is an odd number. exit')
    import sys; sys.exit()

# print out ppp
eigs = [0.003465, 0.007274, 0.014955] #maximum frequency for ref 3-5
eig = eigs[ref_level-3]
# if we are doing exp(tL) then minimum period is period of exp(it*eig)
# which is 2*pi/eig
min_period = 2*np.pi/eig
print("ds steps per min wavelength", min_period/dts)
print("dt steps per min wavelength", min_period/dt*nt)

if args.check:
    import sys; sys.exit()

# solver parameters
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

luparams = {
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps'
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
    #"patch_sub_pc_factor_shift_type": "nonzero",
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
    #"patch_sub_pc_factor_shift_type": "nonzero",
}

atol = 1e-10
rtol = 1e-8
solver_parameters_diag = {
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'monitor': None,
        'converged_reason': None,
        'rtol': rtol,
        'atol': atol,
        'stol': 1e-12,
    },
    'pc_type': 'python',
    'pc_python_type': 'asQ.DiagFFTPC',
    'diagfft_alpha': args.alphap,
    'diagfft_state': 'linear',
    'aaos_jacobian_state': 'linear',
}

# setup parameters for paradiag solve (the number of windows is fixed to 1)
solver_parameters_diag['diagfft_block_'+str(i)+'_'] = monoparameters_ns

PETSc.Sys.Print('### === --- Calculating parallel solution --- === ###')

# setup slice length
slice_length_t = int(nt/args.nslices)
assert(slice_length_t*args.nslices == nt)
slice_length_s = int(ns/args.nslices)
assert(slice_length_s*args.nslices == ns)

# setup time_partition
time_partition_t = [slice_length_t for _ in range(args.nslices)]
time_partition_s = [slice_length_s for _ in range(args.nslices)]

# setup ensemble
ensemble = asQ.create_ensemble(time_partition_t)
ensemble_rank = ensemble.ensemble_comm.rank

#some domain, parameters and FS setup
R0 = 6371220.
H0 = Constant(5960.)

mesh_degree = 3
mesh = IcosahedralSphereMesh(radius=R0,
                             refinement_level=ref_level, degree=mesh_degree, comm=ensemble.comm)
cx = SpatialCoordinate(mesh)
mesh.init_cell_orientations(cx)

cx, cy, cz = SpatialCoordinate(mesh)

outward_normals = interpolate(CellNormal(mesh),VectorFunctionSpace(mesh,"DG",mesh_degree))
DG2 = VectorFunctionSpace(mesh, "DG", 2)
outward_normals_appx = Function(DG2).interpolate(outward_normals)
perp = lambda u: cross(outward_normals_appx, u)
    
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

#Set up all at once functions
Wall = asQ.AllAtOnceFunction(ensemble, time_partition_t, W)
Walls = asQ.AllAtOnceFunction(ensemble, time_partition_t, W)
Nall = asQ.AllAtOnceFunction(ensemble, time_partition_s, W)
Xall = asQ.AllAtOnceFunction(ensemble, time_partition_s, W)
RHS = asQ.AllAtOnceFunction(ensemble, time_partition_s, W)
RHS_new = asQ.AllAtOnceFunction(ensemble, time_partition_s, W)

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
    constant_jacobian = False
else:
    constant_jacobian = True

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
    + phi*(D - D0) + dt_ss*div(H*uh)*phi
)*dx

if args.advection:
    F1p += dt_ss*advection(uh, ubar, v, vector=True)
    F1p += dt_ss*advection(Dh, ubar, phi,
                           continuity=True, vector=False)
    
u0, D0 = split(W0)    
    
uh = (u+u1)/2
Dh = (D+D1)/2
# positive s inward propagation
F0p = (
    inner(v, u1 - u) + dt_ss*inner(f*perp(uh),v) - dt_ss*g*Dh*div(v)
    + phi*(D1 - D) + dt_ss*div(uh*H)*phi
)*dx

if args.advection:
    F0p += dt_ss*advection(uh, ubar, v, vector=True, upwind=False)
    F0p += dt_ss*advection(Dh, ubar, phi, vector=False,
                           continuity=True, upwind=False)

dt_ss = -dt_s
# negative s outward  propagation
F1m = (
    inner(v, u - u0) + dt_ss*inner(f*perp(uh),v) - dt_ss*g*Dh*div(v)
    + phi*(D - D0) + dt_ss*div(uh*H)*phi
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
    + phi*(D1 - D) + dt_ss*div(uh*H)*phi
)*dx

if args.advection:
    F0m += dt_ss*advection(uh, ubar, v, vector=True)
    F0m += dt_ss*advection(Dh, ubar, phi, continuity=True, vector=False)

#if args.advection:
#    params = monoparameters_ns
#else:
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
#if args.advection:
#    params = monoparameters_nt
#else:
params = hparams

u, D = TrialFunctions(W)
uh = (u0+u)/2
Dh = (D0+D)/2

dt_ss = Constant(dt/nt)
# positive s outward propagation
F1p = (
    inner(v, u - u0) + dt_ss*inner(f*perp(uh),v) - dt_ss*g*Dh*div(v)
    + phi*(D - D0) + dt_ss*div(uh*H)*phi
)*dx

if args.advection:
    F1p += dt_ss*advection(uh, ubar, v, vector=True)
    F1p += dt_ss*advection(Dh, ubar, phi,
                           continuity=True, vector=False)

forwardp_expProb_dt = LinearVariationalProblem(lhs(F1p), rhs(F1p), W1,
                                            constant_jacobian=constant_jacobian)
forwardp_expsolver_dt = LinearVariationalSolver(forwardp_expProb_dt,
                                                solver_parameters=params)

### === --- Set up form_function and form_mass for ParaDiag --- === ###
def get_form_function(upwind=True):
    def form_function(u, D, v, phi, t):
        F1p = (inner(f*perp(u),v) - g*D*div(v) + div(u*H)*phi)*dx

        if args.advection:
            F1p += advection(u, ubar, v, vector=True, upwind=upwind)
            F1p += advection(D, ubar, phi,
                             continuity=True, vector=False, upwind=upwind)
        return F1p
    return form_function

def form_mass(uu, up, vu, vp):
    return (inner(uu, vu) + up * vp) * dx

theta = Constant(0.5)

### === --- Set up the forward solver for dt propagation --- === ###
propagate_form = asQ.AllAtOnceForm(Wall, dt/nt, theta,
                                   form_mass, get_form_function())
propagate_solver = asQ.AllAtOnceSolver(propagate_form, Wall,
                                       solver_parameters=solver_parameters_diag)
propagate_form_half = asQ.AllAtOnceForm(Wall, 0.5*dt/nt, theta,
                                        form_mass, get_form_function())
propagate_solver_half = asQ.AllAtOnceSolver(propagate_form_half, Wall,
                                            solver_parameters=solver_parameters_diag)

### === --- Set up the forward scatter solver for ns propagation --- === ###
Wpform = asQ.AllAtOnceForm(Walls, alpha*dt/ns, theta,
                           form_mass, get_form_function(upwind=True))
Wpsolver = asQ.AllAtOnceSolver(Wpform, Walls,
                               solver_parameters=solver_parameters_diag)
Wmform = asQ.AllAtOnceForm(Walls, -alpha*dt/ns, theta,
                           form_mass, get_form_function(upwind=False))
Wmsolver = asQ.AllAtOnceSolver(Wmform, Walls,
                               solver_parameters=solver_parameters_diag)

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
# Hack to unify code when no nonlinearity
Zero = Function(V2).assign(0.)
L += phi*Zero*dx
if not args.eta:
    L -= div(v)*g*b*dx

if not args.linear_velocity:
    if vector_invariant:
        L -= (
            + inner(perp(grad(inner(v, perp(u1)))), u1)*dx
            - inner(both(perp(n)*inner(v, perp(u1))),
                    both(Upwind*u1))*dS
            + div(v)*K*dx
        )
    else:
        L += advection(u1, u1, v, vector=True)
if not args.linear_height:
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

if args.advection:
    if not args.linear_velocity:
        L -= advection(u1, ubar, v, vector=True)
    if not args.linear_height:
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
uh = (1-theta)*w_k*nu
Dh = (1-theta)*w_k*nD

dt_ss = dt_s
# positive s inward propagation
Ftp = (
    dt_ss*inner(f*perp(uh),v) - dt_ss*g*Dh*div(v)
    + dt_ss*div(uh*H)*phi
)*dx
Ftp += (inner(v, w_k*nu) + phi*w_k*nD)*dx
if args.advection:
    # we are going backwards in time
    Ftp += dt_ss*advection(uh, ubar, v, upwind=True, vector=True)
    Ftp += dt_ss*advection(Dh, ubar, phi, continuity=True,
                           upwind=True, vector=False)

dt_ss = -dt_s
# negative s inward propagation
Ftm = (
    dt_ss*inner(f*perp(uh),v) - dt_ss*g*Dh*div(v)
    + dt_ss*div(uh*H)*phi
)*dx
Ftm += (inner(v, w_k*nu) + phi*w_k*nD)*dx
if args.advection:
    Ftm += dt_ss*advection(uh, ubar, v, vector=True, upwind=False)
    Ftm += dt_ss*advection(Dh, ubar, phi, continuity=True, vector=False, upwind=False)


Xpform = asQ.AllAtOnceForm(Xall, alpha*dt/ns, theta,
                           form_mass, get_form_function(upwind=True))
Xpsolver = asQ.AllAtOnceSolver(Xpform, Xall,
                               solver_parameters=solver_parameters_diag)
Xmform = asQ.AllAtOnceForm(Xall, -alpha*dt/ns, theta,
                           form_mass, get_form_function(upwind=False))
Xmsolver = asQ.AllAtOnceSolver(Xmform, Xall,
                               solver_parameters=solver_parameters_diag)

# total number of points is 2ns + 1, because we have s=0
# after forward loop, W1 contains value at time ns*ds
# if we start with X^{ns+1}=0, then according to above
# (1 + ds/2*L)X^{ns} = w_k(1+ds/2*L)N(W_{ns})
# which is equivalent to X^{ns} = N(W_{ns})
# so everything is working
# we just need to change the order to
# compute N, use to propagate X back, propagate W back
# don't need to propagate W back on last step though

# true svals goes from -rho*dt/2 to rho*dt/2
# this is shifted to [0,1] and only compute the second half
svals = 0.5 + np.arange(ns)/ns/2
# don't include 1 because we'll get NaN
weights = np.exp(-1.0/svals/(1.0-svals))
# half the 0 point because it is counted twice
# once for positive s and once for negative s
weights[0] /= 2
# renormalise and then half because once for each sign
weights = weights/np.sum(weights)/2
# include a 0 on the end
weights = np.concatenate((weights, [0]))

print(weights)

offset_list = []
for i_rank in range(len(time_partition_s)):
    offset_list.append(sum(time_partition_s[:i_rank]))

def index2rank(index):
    for rank in range(len(offset_list)):
        if offset_list[rank] - index > 0:
            rank -= 1
            break
    return rank

def data_flip(Nall, Nall_new):

    mpi_requests = []

    for ilocal in range(time_partition_s[ensemble_rank]):
        iglobal = Nall.transform_index(ilocal, from_range='slice', to_range='window')
        target = ns-iglobal-1

        request_send = ensemble.isend(
            Nall[ilocal],
            dest=index2rank(target),
            tag=target)
        mpi_requests.extend(request_send)

        request_recv = ensemble.irecv(
            Nall_new[ilocal],
            source=index2rank(target),
            tag=iglobal)
        mpi_requests.extend(request_recv)

    MPI.Request.Waitall(mpi_requests)

Fall = asQ.AllAtOnceFunction(ensemble, time_partition_s, V2)
Fall_new = asQ.AllAtOnceFunction(ensemble, time_partition_s, V2)

for ilocal in range(time_partition_s[ensemble_rank]):
    iglobal = Fall.transform_index(ilocal, from_range='slice', to_range='window')
    Fall[ilocal].assign(iglobal)

data_flip(Fall, Fall_new)

for ilocal in range(time_partition_s[ensemble_rank]):
    iglobal = Fall_new.transform_index(ilocal, from_range='slice', to_range='window')
    i = ns-iglobal-1
    assert(norm(Fall_new[ilocal] - i) < 1.e-08)

# Function to take in current state V and return dV/dt
def average(V, dVdt, positive=True, t=None):
    # forward scatter
    Walls.assign(V)
    with PETSc.Log.Event("forward propagation ds"):
        if positive:
            Wpsolver.solve()
        else:
            Wmsolver.solve()
    Walls.bcast_field(-1, W1)

    # backwards gather
    for step in range(time_partition_s[ensemble_rank]):
        W1.assign(Walls[step])
        step_W = Walls.transform_index(step, from_range='slice', to_range='window')
        w_k.assign(weights[step_W+1])
        NSolver.solve()
        if positive:
            Ftmp = assemble(Ftp)
        else:
            Ftmp = assemble(Ftm)
        for rdat,fdat in zip(RHS[step].dat, Ftmp.dat):
            rdat.data[:] = fdat.data[:]
        Nall[step].assign(N)

    Xall.zero()
    # flip the data in Nall
    data_flip(RHS, RHS_new)

    # solve allatoncesolver with RHS in the option
    if positive:
        Xpsolver.solve(rhs=RHS_new)
    else:
        Xmsolver.solve(rhs=RHS_new)
    Xall.bcast_field(-1, dVdt)

    # copy contents
    dVdt.assign(X0)
    

# Function to apply forward propagation in t
def propagate(V_in, V_out, t=None, half=False):
    Wall.assign(V_in)
    if half:
        propagate_solver_half.solve()
    else:
        propagate_solver.solve()
    Wall.bcast_field(-1, V_out)


t = 0.
tmax = 60.*60.*args.tmax
#tmax = dt
dumpt = args.dumpt*60.*60.
checkt = args.checkt*60.*60.
tdump = 0.
tcheck = 0.

x = SpatialCoordinate(mesh)

# Topography
rl = pi/9.0
lambda_x = atan2(x[1]/R0, x[0]/R0)
lambda_c = -pi/2.0
phi_x = asin(x[2]/R0)
phi_c = pi/6.0
minarg = min_value(pow(rl, 2), pow(phi_x - phi_c, 2) + pow(lambda_x - lambda_c, 2))
bexpr = 2000.0*(1 - sqrt(minarg)/rl)
b.interpolate(bexpr)

# set u_expr, eta_expr
u_0 = 20.0  # maximum amplitude of the zonal wind [m/s]
u_max = Constant(u_0)
u_expr = as_vector([-u_max*x[1]/R0, u_max*x[0]/R0, 0.0])
eta_expr = - ((R0 * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R0*R0)))/g

# set un, etan, Dn
un = Function(V1, name="Velocity").project(u_expr)
etan = Function(V2, name="Elevation").interpolate(eta_expr)
if args.eta:
    Dn = Function(V2, name="Elevation").interpolate(eta_expr)
else:
    Dn = Function(V2, name="Depth").interpolate(eta_expr + H - b)

# set ubar
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

#pickup the result
if args.pickup:
    chkfile = DumbCheckpoint(args.pickup_from, mode=FILE_READ)
    chkfile.load(un)
    chkfile.load(etan)
    t = chkfile.read_attribute("/", "time")
    tdump = chkfile.read_attribute("/", "tdump")
    tcheck = chkfile.read_attribute("/", "tcheck")
    chkfile.close()
    if args.eta:
        Dn.assign(etan)
    else:
        Dn.assign(etan+H0-b)
    if args.dynamic_ubar:
        projection_solver.solve()
        u, _ = Uproj.subfunctions
        ubar.assign(un + u)

# write out initial fields
name = args.filename
file_sw = File(name+'.pvd', mode="a")
if not args.pickup:
    file_sw.write(un, etan, b)

# calculate norms for debug
uini = Function(V1, name="Velocity0").project(u_expr)
etaini = Function(V2, name="Elevation0").interpolate(eta_expr)
etanorm = errornorm(etan, etaini)/norm(etaini)
unorm = errornorm(un, uini, norm_type="Hdiv")/norm(uini, norm_type="Hdiv")
print('etanorm', etanorm, 'unorm', unorm)

##############################################################################
# Time loop
##############################################################################
U0 = Function(W)
U1 = Function(W)
U2 = Function(W)
U3 = Function(W)
Ustar = Function(W)
Average = Function(W)
V1 = Function(W)
V2 = Function(W)
V3 = Function(W)
V = Function(W)

# set up initial conditions
U_u, U_D = U0.subfunctions
U1_u, U1_D = U1.subfunctions
U_u.assign(un)
U_D.assign(Dn)

mass0 = assemble(U_D*dx)


#start time loop
print ('tmax', tmax, 'dt', dt)
while t < tmax - 0.5*dt:
    print(t)
    t += dt
    tdump += dt
    tcheck += dt

    if timestepping == 'heuns':
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
        if not args.linear:
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
    elif timestepping == 'rk4':
        print("RK stage 1")
        average(U0, Average, positive=True, t=t)
        V1.assign(dt*Average)
        average(U0, Average, positive=False, t=t)
        V1 += dt*Average
        U1.assign(U0 + V1/2)

        print("RK stage 2")
        propagate(U1, U1, t=t, half=True)
        average(U1, Average, positive=True, t=t)
        V2.assign(dt*Average)
        average(U1, Average, positive=False, t=t)
        V2 += dt*Average
        propagate(U0, V, t=t, half=True)
        U2.assign(V + V2/2)

        print("RK stage 3")
        average(U2, Average, positive=True, t=t)
        V3.assign(dt*Average)
        average(U2, Average, positive=False, t=t)
        V3 += dt*Average
        U3.assign(V + V3)

        print("RK stage 4")
        U1.assign(V2 + V3)
        propagate(U1, U2, t=t, half=True)
        propagate(U3, V3, t=t, half=True)
        average(V3, Average, positive=True, t=t)
        U3.assign(dt*Average)
        average(V3, Average, positive=False, t=t)
        U3 += dt*Average
        propagate(V1, U1, t=t)
        propagate(U0, V, t=t)
        U0.assign(V + 1/6*U1 + 1/3*U2 + 1/6*U3)

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

    #create checkpointing file every tcheck hours
    if tcheck > checkt - dt*0.5:
        un.assign(U_u)
        Dn.assign(U_D)
        if args.eta:
            etan.assign(Dn)
        else:
            etan.assign(Dn-H0+b)
        thours = int(t/3600)
        chk = DumbCheckpoint(name+"_"+str(thours)+"h", mode=FILE_CREATE)
        tcheck -= checkt
        chk.store(un)
        chk.store(etan)
        chk.write_attribute("/", "time", t)
        chk.write_attribute("/", "tdump", tdump)
        chk.write_attribute("/", "tcheck", tcheck)
        chk.close()
        print("checkpointed at t =", t)

        #calculate norms for debug
        etanorm = errornorm(etan, etaini)/norm(etaini)
        unorm = errornorm(un, uini, norm_type="Hdiv")/norm(uini, norm_type="Hdiv")
        print('etanorm', etanorm, 'unorm', unorm)

print("Completed calculation at t = ", t/3600, "hours")
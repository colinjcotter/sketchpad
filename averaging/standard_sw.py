from firedrake import *
from math import pi
from math import ceil
from firedrake.petsc import PETSc
from firedrake.__future__ import interpolate
import numpy as np
import argparse
print = PETSc.Sys.Print

#get command arguments
parser = argparse.ArgumentParser(description='Williamson 5 testcase for averaged propagator.')
parser.add_argument('--ref_level', type=int, default=3, help='Refinement level of icosahedral grid. Default 4.')
parser.add_argument('--tmax', type=float, default=360, help='Final time in hours. Default 24x15=360.')
parser.add_argument('--dumpt', type=float, default=24, help='Dump time in hours. Default 24.')
parser.add_argument('--checkt', type=float, default=6, help='Create checkpointing file every checkt hours. Default 6.')
parser.add_argument('--dt', type=float, default=1800, help='Timestep for the standard model in seconds. Default 45.')
parser.add_argument('--filename', type=str, default='standard')
parser.add_argument('--pickup', action='store_true', help='Pickup the result from the checkpoint.')

args = parser.parse_known_args()
args = args[0]
ref_level = args.ref_level
name = args.filename
dt = args.dt
print(args)

#parameters
R0 = 6371220.
R = Constant(R0)
H = Constant(5960.)
Omega = Constant(7.292e-5)  # rotation rate
g = Constant(9.8)  # Gravitational constant
mesh_degree = 3

if args.pickup:
    # pickup the mesh when --pickup is specified
    with CheckpointFile(name+".h5", 'r') as checkpoint:
        mesh = checkpoint.load_mesh("mesh")
        x = SpatialCoordinate(mesh)
        print("Picked up the mesh from "+name+".h5")
else:
    #create mesh
    mesh = IcosahedralSphereMesh(radius=R0, refinement_level=ref_level,
                                 degree=mesh_degree, name="mesh")
    x = SpatialCoordinate(mesh)
    global_normal = as_vector([x[0], x[1], x[2]])
    mesh.init_cell_orientations(global_normal)
    print("Created mesh")

outward_normals = interpolate(CellNormal(mesh),VectorFunctionSpace(mesh,"DG",mesh_degree))
perp = lambda u: cross(outward_normals, u)
V1 = FunctionSpace(mesh, "BDM", 2)
V2 = FunctionSpace(mesh, "DG", 1)
W = MixedFunctionSpace((V1, V2))
f_expr = 2 * Omega * x[2] / R
Vf = FunctionSpace(mesh, "CG", 3)
f = Function(Vf).interpolate(f_expr)    # Coriolis frequency
u_0 = 20.0  # maximum amplitude of the zonal wind [m/s]
u_max = Constant(u_0)
u_expr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
eta_expr = -((R*Omega*u_max+u_max*u_max/2.0)*(x[2]*x[2]/(R*R)))/g
h_expr = eta_expr + H

#topography (D = H + eta - b)
rl = pi/9.0
lambda_x = atan2(x[1]/R0, x[0]/R0)
lambda_c = -pi/2.0
phi_x = asin(x[2]/R0)
phi_c = pi/6.0
minarg = min_value(pow(rl, 2), pow(phi_x - phi_c, 2) + pow(lambda_x - lambda_c, 2))
bexpr = 2000.0*(1 - sqrt(minarg)/rl)
b = Function(V2, name="Topography")
b.interpolate(bexpr)

#parameters for timestepping
t = 0.
tmax = 60.*60.*args.tmax
dumpt = args.dumpt*60.*60.
checkt = args.checkt*60.*60.
tdump = 0.
tcheck = 0.
idx = 0
k_max = 4        # Maximum number of Picard iterations

#initiliase un, etan, hn
un = Function(V1, name="Velocity").project(u_expr)
etan = Function(V2, name="Elevation").interpolate(eta_expr)
hn = Function(V2, name="Depth").interpolate(h_expr)
hn -= b

#set uini, etaini for debug
uini = Function(V1, name="Velocity0").project(u_expr)
etaini = Function(V2, name="Elevation0").interpolate(eta_expr)

#setup PV solver
PV = Function(Vf, name="PotentialVorticity")
gamma = TestFunction(Vf)
q = TrialFunction(Vf)
D = hn
a = q*gamma*D*dx
L = (- inner(perp(grad(gamma)), un))*dx + gamma*f*dx
PVproblem = LinearVariationalProblem(a, L, PV)
PVsolver = LinearVariationalSolver(PVproblem, solver_parameters={"ksp_type": "cg"})
PVsolver.solve()

file_sw = output.VTKFile(name+'.pvd', mode="a")
if args.pickup:
    # pickup the result when --pickup is specified
    with CheckpointFile(name+".h5", 'r') as checkpoint:
        timestepping_history = checkpoint.get_timestepping_history(mesh, name="Velocity")
        idx = len(timestepping_history["index"]) - 1
        PETSc.Sys.Print("Picking up the result from ", name+".h5,", "last index is", idx)
        un = checkpoint.load_function(mesh, "Velocity", idx=idx)
        etan = checkpoint.load_function(mesh, "Elevation", idx=idx)
        t = timestepping_history["time"][-1]
        tdump = timestepping_history["tdump"][-1]
        tcheck = timestepping_history["tcheck"][-1]
        print("Restart from t = ", t, ", tdump = ", tdump, ", tcheck = ", tcheck)
    hn.assign(etan + H - b)
else:
    # dump initial condition when --pickup is not specified
    file_sw.write(un, etan, PV, b)
    # create checkpoint file and save initial condition
    with CheckpointFile(name+".h5", 'w') as checkpoint:
        checkpoint.save_mesh(mesh)
        checkpoint.save_function(b)
        checkpoint.save_function(un, idx=idx,
                                 timestepping_info={"time": t, "tdump": tdump, "tcheck": tcheck})
        checkpoint.save_function(etan, idx=idx,
                                 timestepping_info={"time": t, "tdump": tdump, "tcheck": tcheck})
        checkpoint.save_function(PV, idx=idx,
                                 timestepping_info={"time": t, "tdump": tdump, "tcheck": tcheck})
        print("Checkpointed at t = ", t, "idx = ", idx)

#calculate norms for debug
etanorm = errornorm(etan, etaini)/norm(etaini)
unorm = errornorm(un, uini, norm_type="Hdiv")/norm(uini, norm_type="Hdiv")
print('etanorm', etanorm, 'unorm', unorm)


##############################################################################
# Set up solvers
##############################################################################
# Set up depth advection solver (DG upwinded scheme)
up = Function(V1)
hp = Function(V2)
hpt = Function(V2)
h = TrialFunction(V2)
phi = TestFunction(V2)
hh = 0.5 * (hn + h)
uh = 0.5 * (un + up)
n = FacetNormal(mesh)
uup = 0.5 * (dot(uh, n) + abs(dot(uh, n)))
Heqn = ((h - hn)*phi*dx - dt*inner(grad(phi), uh*hh)*dx
        + dt*jump(phi)*(uup('+')*hh('+')-uup('-')*hh('-'))*dS)
Hproblem = LinearVariationalProblem(lhs(Heqn), rhs(Heqn), hpt)
lu_params = {'ksp_type': 'preonly',
             'pc_type': 'lu',
             'pc_factor_mat_solver_type': 'mumps'}
Hsolver = LinearVariationalSolver(Hproblem,
                                  solver_parameters=lu_params,
                                  options_prefix="H-advection")

# Velocity advection (Natale et. al (2016) extended to SWE)
upt = Function(V1)
u = TrialFunction(V1)
v = TestFunction(V1)
hh = 0.5 * (hn + hp)
ubar = 0.5 * (un + up)
uup = 0.5 * (dot(ubar, n) + abs(dot(ubar, n)))
uh = 0.5 * (un + u)
Upwind = 0.5 * (sign(dot(ubar, n)) + 1)
K = 0.5 * (inner(0.5 * (un + up), 0.5 * (un + up)))
both = lambda u: 2*avg(u)
#outward_normals = CellNormal(mesh)
perp = lambda arg: cross(outward_normals, arg)
Ueqn = (inner(u - un, v)*dx + dt*inner(perp(uh)*f, v)*dx
        - dt*inner(perp(grad(inner(v, perp(ubar)))), uh)*dx
        + dt*inner(both(perp(n)*inner(v, perp(ubar))),
                   both(Upwind*uh))*dS
        - dt*div(v)*(g*(hh + b) + K)*dx)
Uproblem = LinearVariationalProblem(lhs(Ueqn), rhs(Ueqn), upt)
Usolver = LinearVariationalSolver(Uproblem,
                                  solver_parameters=lu_params,
                                  options_prefix="U-advection")

# Linear solver for incremental updates
HU = Function(W)
deltaU, deltaH = HU.subfunctions
w, phi = TestFunctions(W)
du, dh = TrialFunctions(W)
alpha = 0.5
HUlhs = (inner(w, du + alpha*dt*f*perp(du))*dx
         - alpha*dt*div(w)*g*dh*dx
         + phi*(dh + alpha*dt*H*div(du))*dx)
HUrhs = -inner(w, up - upt)*dx - phi*(hp - hpt)*dx
HUproblem = LinearVariationalProblem(HUlhs, HUrhs, HU)
params = {'ksp_type': 'preonly',
          'mat_type': 'aij',
          'pc_type': 'lu',
          'pc_factor_mat_solver_type': 'mumps'}
HUsolver = LinearVariationalSolver(HUproblem,
                                   solver_parameters=params,
                                   options_prefix="impl-solve")

##############################################################################
# Time loop
##############################################################################
#start time loop
print('tmax', tmax, 'dt', dt)
while t < tmax - 0.5*dt:
    print("at the start of time step at t = ", t, ", dt = ", dt, ", tmax = ", tmax)
    t += dt
    tdump += dt
    tcheck += dt

    up.assign(un)
    hp.assign(hn)

    #start picard cycle
    for i in range(k_max):
        #advect to get candidates
        Hsolver.solve()
        Usolver.solve()

        #linear solve for updates
        HUsolver.solve()

        #increment updates
        up += deltaU
        hp += deltaH

    #update fields for next time step
    un.assign(up)
    hn.assign(hp)
    etan.assign(hn + b - H)

    #calculate norms for debug
    etanorm = errornorm(etan, etaini)/norm(etaini)
    unorm = errornorm(un, uini, norm_type="Hdiv")/norm(uini, norm_type="Hdiv")
    print('etanorm', etanorm, 'unorm', unorm)

    #dumping results
    if tdump > dumpt - dt*0.5:
        #dump results
        PVsolver.solve()
        file_sw.write(un, etan, PV, b)
        tdump -= dumpt
        print("dumped results at t =", t)

    #create checkpointing file every tcheck hours
    if tcheck > checkt - dt*0.5:
        tcheck -= checkt
        idx += 1
        PVsolver.solve()
        with CheckpointFile(name+".h5", 'a') as checkpoint:
            checkpoint.save_function(un, idx=idx,
                                     timestepping_info={"time": t, "tdump": tdump, "tcheck": tcheck})
            checkpoint.save_function(etan, idx=idx,
                                     timestepping_info={"time": t, "tdump": tdump, "tcheck": tcheck})
            checkpoint.save_function(PV, idx=idx,
                                     timestepping_info={"time": t, "tdump": tdump, "tcheck": tcheck})
        print("Checkpointed at t = ", t, "idx = ", idx)

print("Completed calculation at t = ", t/3600, "hours")

print("Test Checkpointing")
testc = assemble(dot(un,un)*dx)
print("testc = ", testc)
etanorm = errornorm(etan, etaini)/norm(etaini)
unorm = errornorm(un, uini, norm_type="Hdiv")/norm(uini, norm_type="Hdiv")
print('etanorm', etanorm, 'unorm', unorm)

#print out timestepping history in the checkpoint file for debug
with CheckpointFile(name+".h5", 'r') as checkpoint:
    mesh = checkpoint.load_mesh("mesh")
    timestepping_history = checkpoint.get_timestepping_history(mesh, name="Velocity")
    print("timestepping_history=", timestepping_history)
    print("timestepping_history_index=", timestepping_history["index"])
    print("timestepping_history_time=", timestepping_history["time"])
    print("timestepping_history_tdump=", timestepping_history["tdump"])
    print("timestepping_history_tcheck=", timestepping_history["tcheck"])
    print("timestepping_history_index_last=", timestepping_history["index"][-1])
    print("timestepping_history_time_last=", timestepping_history["time"][-1])
    print("timestepping_history_tdump_last=", timestepping_history["tdump"][-1])
    print("timestepping_history_tcheck_last=", timestepping_history["tcheck"][-1])

    uini = checkpoint.load_function(mesh, "Velocity", idx=0)
    etaini = checkpoint.load_function(mesh, "Elevation", idx=0)
    for i in range(len(timestepping_history["time"])):
        un = checkpoint.load_function(mesh, "Velocity", idx=i)
        etan = checkpoint.load_function(mesh, "Elevation", idx=i)
        print("Picked up at idx = ", i)

        testc = assemble(dot(un,un)*dx)
        print("testc = ", testc)

        etanorm = errornorm(etan, etaini)/norm(etaini)
        unorm = errornorm(un, uini, norm_type="Hdiv")/norm(uini, norm_type="Hdiv")
        print('etanorm', etanorm, 'unorm', unorm)

from firedrake import *

m = PeriodicUnitIntervalMesh(100)
T = 1.
nsteps = 10
mesh = ExtrudedMesh(m, 1, 1./nsteps)

degree = 2
CGk_ele = FiniteElement("CG", interval, degree)
DGkm1_ele = FiniteElement("DG", interval, degree-1)
DG0_ele = FiniteElement("DG", interval, 0)
V_ele = TensorProductElement(CGk_ele, CGk_ele)
Q_ele = TensorProductElement(DGkm1_ele, CGk_ele)
Vt_ele = TensorProductElement(CGk_ele, DGkm1_ele)
Qt_ele = TensorProductElement(DGkm1_ele, DGkm1_ele)

DG0_ele = FiniteElement("DG", interval, 0)
V0_ele = TensorProductElement(CGk_ele, DG0_ele)
Q0_ele = TensorProductElement(DGkm1_ele, DG0_ele)

V = FunctionSpace(mesh, V_ele)
Q = FunctionSpace(mesh, Q_ele)
Vt = FunctionSpace(mesh, Vt_ele)
Qt = FunctionSpace(mesh, Qt_ele)
Vr = RestrictedFunctionSpace(V, boundary_set=["bottom"])
Qr = RestrictedFunctionSpace(Q, boundary_set=["bottom"])
W = Vr * Qr
Wt = Vt * Qt

V0 = FunctionSpace(mesh, V0_ele)
Q0 = FunctionSpace(mesh, Q0_ele)

u0 = Function(V0)
p0 = Function(Q0)

x, t = SpatialCoordinate(mesh)

sigma = 0.1
p0.interpolate(exp(-(x-0.5)**2/sigma**2))

w = Function(W)
v, q = TestFunctions(Wt)
u, p = TrialFunctions(W)

u = u0 + u
p = p0 + p

eqn = (
    v*u.dx(1)*dx - v.dx(0)*p*dx
    + q*p.dx(1)*dx + q*u.dx(0)*dx
    )

u, p = split(w)
v = TestFunction(V0)
u_next = TrialFunction(V0)
v_update_eqn = (
    v*(u_next - u - u0)*ds_t
)

q = TestFunction(Q0)
p_next = TrialFunction(Q0)
p_update_eqn = (
    q*(p_next - p - p0)*ds_t
)

outfile = VTKFile("outfile_mixed.pvd")
outfile.write(u0, p0)

step_problem = LinearVariationalProblem(lhs(eqn), rhs(eqn), w)
stepper = LinearVariationalSolver(step_problem)

u_update_problem = LinearVariationalProblem(lhs(v_update_eqn),
                                            rhs(v_update_eqn), u0)
u_update_solver = LinearVariationalSolver(u_update_problem)
p_update_problem = LinearVariationalProblem(lhs(p_update_eqn),
                                            rhs(p_update_eqn), p0)
p_update_solver = LinearVariationalSolver(p_update_problem)

nsteps = 10
for step in ProgressBar("Timestep").iter(range(nsteps)):
    stepper.solve()
    # update initial condition
    u_update_solver.solve()
    p_update_solver.solve()
    
    outfile.write(u0, p0)

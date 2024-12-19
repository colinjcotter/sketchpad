from firedrake import *

n = 20
m = PeriodicUnitSquareMesh(n, n)
T = 1.
nsteps = 10
mesh = ExtrudedMesh(m, 1, 1./nsteps)

tdegree = 1
sdegree = 1
BDMk_ele = FiniteElement("BDM", triangle, sdegree)
DGkm1_ele = FiniteElement("DG", triangle, sdegree-1)
CGl_ele = FiniteElement("DG", interval, tdegree)
DGlm1_ele = FiniteElement("DG", interval, tdegree-1)

V_ele = HDiv(TensorProductElement(BDMk_ele, CGl_ele))
Q_ele = TensorProductElement(DGkm1_ele, CGl_ele)
Vt_ele = HDiv(TensorProductElement(BDMk_ele, DGlm1_ele))
Qt_ele = TensorProductElement(DGkm1_ele, DGlm1_ele)

DG0_ele = FiniteElement("DG", interval, 0)
V0_ele = HDiv(TensorProductElement(BDMk_ele, DG0_ele))
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

x, y, t = SpatialCoordinate(mesh)

sigma = 0.1
p0.interpolate(exp(-((x-0.5)**2 + (y-0.5)**2)/sigma**2))

w = Function(W)
v, q = TestFunctions(Wt)
deltau, deltap = TrialFunctions(W)

peqn = (
    inner(v,deltau)*dx + div(v)*div(deltau)*dx
    + q*deltap*dx
    )

u = u0 + deltau
p = p0 + deltap

def div(u):
    return u[0].dx(0) + u[1].dx(1)

eqn = (
    inner(v,u.dx(2))*dx - div(v)*p*dx
    + q*p.dx(2)*dx + q*div(u)*dx
    )

deltau, deltap = split(w)
v = TestFunction(V0)
u_next = TrialFunction(V0)
v_update_eqn = (
    inner(v,(u_next - deltau - u0))*ds_t
)

q = TestFunction(Q0)
p_next = TrialFunction(Q0)
p_update_eqn = (
    q*(p_next - deltap - p0)*ds_t
)

outfile = VTKFile("outfile_mixed.pvd")
outfile.write(u0, p0)

params = {
    "ksp_type": "gmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "additive",
    "fieldsplit_0_pc_type": "lu",
    "fieldsplit_1_pc_type": "lu",
    "ksp_monitor": None
    }

step_problem = LinearVariationalProblem(lhs(eqn), rhs(eqn), w,
                                        aP=peqn)
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

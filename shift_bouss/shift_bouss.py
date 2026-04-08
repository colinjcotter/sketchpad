from firedrake import *

nz = 100
nx = 100
length = 3.0e5
height = 1.0e3
dt = Constant(10000)
shift = Constant(1.0e-1)
Nsq = Constant(1.0e-4)

m = PeriodicIntervalMesh(nx, length)
mesh = ExtrudedMesh(m, nz, layer_height=height/nz)

x, z = SpatialCoordinate(mesh)
deg = 1
CG = FiniteElement("CG", interval, deg)
DG = FiniteElement("DG", interval, deg-1)
CG_DG = TensorProductElement(CG, DG)
RT_horiz = HDivElement(CG_DG)
DG_CG = TensorProductElement(DG, CG)
RT_vert = HDivElement(DG_CG)
RT_e = RT_horiz + RT_vert
V_2D = FunctionSpace(mesh, RT_e)
DG_km1 = FunctionSpace(mesh, 'DG', deg-1)
Pressure = DG_km1
horiz = FiniteElement('DG', interval, deg-1)
vertical = FiniteElement('CG', interval, deg)
V_elt = TensorProductElement(horiz, vertical)
Vb = FunctionSpace(mesh, V_elt)
W = V_2D * Vb * Pressure

U0 = Function(W)
u0, b0, p0 = U0.subfunctions
xc = Constant(length/2)
yc = Constant(length/2)
a = Constant(5000)
b0.interpolate(sin(pi*z/height)/(1+((x-xc)**2)/a**2))
u0.project(as_vector([0, exp(-(x-xc)**2/a)]))
k = as_vector([0,1])

u, b, p = TrialFunctions(W)
w, gam, q = TestFunctions(W)
V = TestFunction(W)

eqn = (
    inner(w, u) - div(w)*p*dt - inner(w,k)*b*dt
    + gam*(b + inner(u, k)*Nsq*dt)
    + q*div(u)*dt
    + inner(V, U0)
    )*dx

J_shift = lhs(eqn + shift*q*p*dx)

U1 = Function(W)

bc1 = DirichletBC(W.sub(0), as_vector([0., 0.]), "top")
bc2 = DirichletBC(W.sub(0), as_vector([0., 0.]), "bottom")
bcs = [bc1, bc2]

problem = LinearVariationalProblem(lhs(eqn), rhs(eqn), U1, aP=J_shift,
                                   bcs=bcs)
params = {
    'ksp_monitor': None,
    'mat_type':'aij',
    'ksp_type': 'gmres',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps'
}
solver = LinearVariationalSolver(problem, solver_parameters=params)
solver.solve()

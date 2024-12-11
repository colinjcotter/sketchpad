from firedrake import *

m = UnitIntervalMesh(10)
mesh = ExtrudedMesh(m, 1)

CG1_ele = FiniteElement("CG", interval, 1)
DG0_ele = FiniteElement("DG", interval, 0)
trial_ele = TensorProductElement(CG1_ele, CG1_ele)
test_ele = TensorProductElement(CG1_ele, DG0_ele)

V = FunctionSpace(mesh, trial_ele)
Vr = RestrictedFunctionSpace(V, boundary_set=["bottom"])
W = FunctionSpace(mesh, test_ele)

u = Function(Vr)
v = TestFunction(W)

bcs = [DirichletBC(Vr, 1.0, ["bottom"])]

eqn = v*u.dx(1)*dx - v*dx

solve(eqn == 0, u)

outfile = VTKFile("outfile.pvd")
outfile.write(u)

from firedrake import *

m = IntervalMesh(10)
dt = 0.1
mesh = ExtrudedMesh(m, 1, dt)

CG1_ele = FiniteElement("CG", interval, 1)
DG0_ele = FiniteElement("DG", interval, 0)
trial_ele = TensorProductElement(CG1_ele, CG1_ele)
test_ele = TensorProductElement(CG1_ele, DG0_ele)

V_trial_unrestricted = FunctionSpace(mesh, trial_ele)
V_trial = RestrictedFunctionSpace(V_trial_unrestricted,
                                  boundary_set="bottom")
V_test = FunctionSpace(mesh, test_ele)

x, t = SpatialCoordinate(mesh)

u0 = Function(V_trial).assign(sin(2*pi*x))

bcs = [DirichletBC(V_trial, u0, "bottom")]

from firedrake import *
mesh = UnitIntervalMesh(20)

V = VectorFunctionSpace(mesh, "R", dim=4)

a = Function(V).assign(1.0)
b = Function(V).assign(2.0)

a.assign(b)

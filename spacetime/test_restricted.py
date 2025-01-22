from firedrake import *

mesh = UnitSquareMesh(10,10)
V = FunctionSpace(mesh, "CG", 1)
Vr = RestrictedFunctionSpace(V, boundary_set=["on_boundary"])

u = Function(Vr)
v = TestFunction(Vr)

bcs = [DirichletBC(Vr, 1.0, "on_boundary")]

eqn = inner(grad(v), grad(u))*dx - v*dx

solve(eqn == 0, u, bcs=bcs)

outfile = VTKFile("outfile.pvd")
outfile.write(u)

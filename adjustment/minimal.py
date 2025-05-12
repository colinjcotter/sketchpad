from firedrake import *

n = 20
mesh = UnitIntervalMesh(n)

V = FunctionSpace(mesh, "CG", 1)

W = V*V

U = Function(W)

u1, u2 = split(U)
x, = SpatialCoordinate(mesh)


du1, du2 = TestFunctions(W)

eqn = (
    du1*(u1-x) + du2*(u2-x+0.5)
)*dx

problem = NonlinearVariationalProblem(eqn, U)

lbound = Function(W).assign(PETSc.NINFINITY)
ubound = Function(W).assign(PETSc.INFINITY)
lbound.sub(1).assign(0)

params = {
    "snes_type": "vinewtonrsls",
    "snes_monitor": None,
    "snes_vi_monitor": None,
    "ksp_type": "preonly",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_0_fields": "0",
    "pc_fieldsplit_1_fields": "1",
    "pc_fieldsplit_type": "additive",
    "snes_atol": 1.0e-8
}

solver = NonlinearVariationalSolver(problem,
                                    solver_parameters=params)


solver.solve(bounds=(lbound, ubound))

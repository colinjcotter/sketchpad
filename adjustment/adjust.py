from firedrake import *

n = 20
mesh = UnitIntervalMesh(n)

V = FunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "DG", 1)

W = Q*V

U0 = Function(W)
U1 = Function(W)

theta0, F0 = U0.subfunctions
theta1, F1 = U1.subfunctions

F_ground = Constant(0.1)
#F1.assign(F_ground)

x, = SpatialCoordinate(mesh)

theta0.interpolate(x)

theta0, F0 = split(U0)
theta1, F1 = split(U1)

Pi = (2-x)**(2./7)

Dt = Constant(0.001)

dtheta, dF = TestFunctions(W)

eqn = (
    Pi*(theta1 - theta0)*dtheta
    + Dt*dtheta*F1.dx(0)
    - theta1*dF.dx(0)
)*dx

bcs = [DirichletBC(W.sub(1), F_ground, 1),
       DirichletBC(W.sub(1), 0, 1)]


problem = NonlinearVariationalProblem(eqn, U1,  bcs=bcs)

lbound = Function(W).assign(PETSc.NINFINITY)
ubound = Function(W).assign(PETSc.INFINITY)
lbound.sub(1).assign(0)

params = {
    "snes_type": "vinewtonrsls",
    "snes_monitor": None,
    "snes_vi_monitor": None,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "snes_atol": 1.0e-8,
    "pc_factor_mat_solver_type": "mumps"
}

solver = NonlinearVariationalSolver(problem,
                                    solver_parameters=params)


theta1, F1 = U1.subfunctions
file0 = VTKFile("output.pvd")
file0.write(theta1)

nsteps = 50
for step in range(nsteps):
    solver.solve(bounds=(lbound, ubound))
    file0.write(theta1)
    U0.assign(U1)

from firedrake import *
n = 50

mesh = UnitSquareMesh(n, n)

V = FunctionSpace(mesh, "CG", 1)
W = V * V

up = Function(W)
up_prev = Function(W)

x, y = SpatialCoordinate(mesh)


obstacle = conditional(ge(x, 0.25), 1.0, 0.0)
obstacle *= conditional(le(x, 0.75), 1.0, 0.0)
obstacle *= conditional(ge(y, 0.25), 1.0, 0.0)
obstacle *= conditional(le(y, 0.75), 1.0, 0.0)

VV = FunctionSpace(mesh, "DG", 1)
obstacle = Function(VV).interpolate(obstacle*1.0e-1)

alpha = Constant(1.0)


u, psi = split(up)
u_prev, psi_prev = split(up_prev)

v, w = TestFunctions(W)
eqn = (
    inner(grad(v), alpha*grad(u))*dx +
    (psi-psi_prev)*v*dx
    + w*(u - exp(psi) - obstacle)*dx
)

bcs = [DirichletBC(W.sub(0), 0., "on_boundary"),
       DirichletBC(W.sub(1), 0., "on_boundary")]
prob = NonlinearVariationalProblem(eqn, up, bcs=bcs)
solver_parameters = {'ksp_type': 'preonly',
                     'pc_type':'lu',
                     'snes_monitor': None,
                     'mat_type': 'aij',
                     'snes_linesearch_type': 'basic',
                     }
solver = NonlinearVariationalSolver(prob, solver_parameters=
                                    solver_parameters)

res = 1.0e50
tol = 1.0e-3

while res > tol:
    solver.solve()
    res = norm(u-u_prev)
    up_prev.assign(up)

    print(res)

u, psi = up.subfunctions
    
VTKFile("obs.pvd").write(u, psi)

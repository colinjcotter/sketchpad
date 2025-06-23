from firedrake import *

n = 100
mesh = UnitSquareMesh(n, n)
V = FunctionSpace(mesh, "CG", 1)
W = V * V

w = Function(W)

u, psi = split(w)

def smooth_abs(u, mu):
    return sqrt(mu**2 + inner(u, u)) - mu

v, phi = TestFunctions(W)

x, y = SpatialCoordinate(mesh)

obstacle = conditional(ge(x, 0.25), 1.0, 0.0)
obstacle *= conditional(le(x, 0.75), 1.0, 0.0)
obstacle *= conditional(ge(y, 0.25), 1.0, 0.0)
obstacle *= conditional(le(y, 0.75), 1.0, 0.0)

eqn = (
    inner(grad(u), grad(v))*dx
    + smooth_abs(grad(u), exp(psi))*v*dx
    - obstacle*v*dx
    + exp(psi)*phi*dx
)

u, psi = w.subfunctions
u.assign(1.0)

bcs = [DirichletBC(W.sub(0), 0., "on_boundary"),
       DirichletBC(W.sub(1), 0., "on_boundary")]

prob = NonlinearVariationalProblem(eqn, w, bcs=bcs)
parameters = {'snes_monitor': None,
              #'ksp_monitor': None,
              'snes_linesearch_type': 'basic',
              'snes_rtol': 1.0e-5}
solver =NonlinearVariationalSolver(prob,
                                   solver_parameters=parameters)

solver.solve()

file0 = VTKFile('max_h1.pvd')
file0.write(u, psi)

from firedrake import *
from firedrake.adjoint import *
from nudging import ensemble_tao_solver

continue_annotation()

ensemble = Ensemble(COMM_WORLD, 2)
rank = ensemble.ensemble_comm.rank
mesh = UnitSquareMesh(20, 20, comm=ensemble.comm)
V = FunctionSpace(mesh, "CG", 1)

n_Js = [3, 3]
Js_offset = [0, 3]
Js = []
Controls = []
xs = []

u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v))*dx

for i in range(n_Js[rank]):
    val = Js_offset[rank]+i+1
    x = function.Function(V)
    L = x*v*dx
    u = Function(V)
    bc = DirichletBC(V, Constant(val), "on_boundary")
    solve(a == L, u, bcs=[bc])
    J = assemble((u*u + x*x)*dx)
    Js.append(J)
    Controls.append(Control(x))
    xs.append(x)
    
Jg_m = []
as1 = []
for i in range(6):
    a = AdjFloat(1.0)
    as1.append(a)
    Jg_m.append(Control(a))
Ja = as1[0]
for i in range(1, 5):
    Ja += as1[i]
Ja = Ja**2
Jg = ReducedFunctional(Ja, Jg_m)
rf = EnsembleReducedFunctional(Js, Controls, ensemble,
                               scatter_control=False,
                               gather_functional=Jg)

stop_annotating()

solver_parameters = {
    "tao_type": "lmvm",
    "tao_cg_type": "pr",
    "tao_monitor": None,
    "tao_converged_reason": None
}

solver = ensemble_tao_solver(rf, ensemble,
                             solver_parameters=solver_parameters)
solver.solve()
solver.tao.view()

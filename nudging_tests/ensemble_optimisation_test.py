from firedrake import *
from firedrake.adjoint import *
from nudging import ensemble_tao_solver
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
import logging
logger = logging.getLogger()
logger.disabled = True
PETSc.Sys.popErrorHandler()

size = MPI.COMM_WORLD.size
ensemble = Ensemble(COMM_WORLD, size//2)
rank = ensemble.ensemble_comm.rank

continue_annotation()
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
    #print("i", i, "val", val, "offset", Js_offset[rank], "ensemble rank", rank)
    x = function.Function(V)
    L = x*v*dx
    u0 = Function(V)
    bc = DirichletBC(V, Constant(val), "on_boundary")
    solve(a == L, u0, bcs=[bc])
    print(rank, norm(u0), norm(x), "NOOORMM", val)
    J = assemble((u0*u0 + x*x)*dx)
    #print("i", i, "Jval", J, "offset", Js_offset[rank], "ensemble rank", rank)
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

jval = rf(xs)
print(jval, "val CJC", ensemble.ensemble_comm.rank,
          ensemble.global_comm.rank)
ders = rf.derivative()
for i, der in enumerate(ders):
    print(norm(der), i, "der CJC", ensemble.ensemble_comm.rank,
          ensemble.global_comm.rank)

solver = ensemble_tao_solver(rf, ensemble,
                             solver_parameters=solver_parameters)
solver.solve()
solver.tao.view()

from firedrake import *
from firedrake.adjoint import *
import pytest
from numpy.testing import assert_allclose
from nudging import ensemble_tao_solver

continue_annotation()

ensemble = Ensemble(COMM_WORLD, 2)
rank = ensemble.ensemble_comm.rank
mesh = UnitSquareMesh(4, 4, comm=ensemble.comm)
R = FunctionSpace(mesh, "R", 0)

n_Js = [2, 3]
Js_offset = [0, 2]
Js = []
Controls = []
xs = []
for i in range(n_Js[rank]):
    val = Js_offset[rank]+i+1
    x = function.Function(R, val=val)
    J = assemble(x * x * dx(domain=mesh))
    Js.append(J)
    Controls.append(Control(x))
    xs.append(x)
    
Jg_m = []
as1 = []
for i in range(5):
    a = AdjFloat(1.0)
    as1.append(a)
    Jg_m.append(Control(a))
Ja = as1[0]**2
for i in range(1, 5):
    Ja += as1[i]**2
Jg = ReducedFunctional(Ja, Jg_m)
val = 1.0**2 + 2.0**2 + 3.0**2 + 4.0**2 + 5.0**2
assert Jg([1., 2., 3., 4., 5.]) == val
rf = EnsembleReducedFunctional(Js, Controls, ensemble,
                               scatter_control=False,
                               gather_functional=Jg)
ensemble_J = rf(xs)

stop_annotating()

solver_parameters = {
    "tao_type": "cg",
    "tao_cg_type": "pr"
}

solver = ensemble_tao_solver(rf, ensemble,
                             solver_parameters=solver_parameters)

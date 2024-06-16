from firedrake import *
from firedrake.adjoint import *
import pytest
from numpy.testing import assert_allclose

continue_annotation()

if True:
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

    if rank == 1:
        assert Js[0] == 9
        assert Js[1] == 16
        assert Js[2] == 25
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
    dJdm = rf.derivative()
    assert_allclose(ensemble_J, 1.0**4+2.0**4+3.0**4+4.0**4+5.0**4, rtol=1e-12)
    perturbations = []
    for i in range(n_Js[rank]):
        val = Js_offset[rank]+i+1
        assert_allclose(dJdm[i].dat.data_ro, 4*val**3, rtol=1e-12)
        perturbations.append(Function(R, val=0.1*i))
    assert taylor_test(rf, xs, perturbations)

from firedrake import *
from irksome import RadauIIA, Dt, MeshConstant, TimeStepper
from ufl.algorithms.ad import expand_derivatives
from irksome.pc import RanaBase
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='IRK Heat equation solver using parallel SDC preconditioner.')
parser.add_argument('--stages', type=int, default=2, help='Number of stages. Default 2.')

args = parser.parse_known_args()
args = args[0]

N = 50

x0 = 0.0
x1 = 10.0
y0 = 0.0
y1 = 10.0

msh = RectangleMesh(N, N, x1, y1)

MC = MeshConstant(msh)
dt = MC.Constant(10. / N)
t = MC.Constant(0.0)

V = FunctionSpace(msh, "CG", 1)
x, y = SpatialCoordinate(msh)

S = Constant(2.0)
C = Constant(1000.0)
B = (x-Constant(x0))*(x-Constant(x1))*(y-Constant(y0))*(y-Constant(y1))/C
R = (x * x + y * y) ** 0.5

uexact = B * atan(t)*(pi / 2.0 - atan(S * (R - t)))
rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))
u = Function(V)
u.interpolate(uexact)

v = TestFunction(V)
F = inner(Dt(u), v)*dx + inner(grad(u), grad(v))*dx - inner(rhs, v) * dx(degree = 4)

bc = DirichletBC(V, 0, "on_boundary")

butcher_tableau = RadauIIA(args.stages)
class PQPC(RanaBase):
    def getAtilde(self, A):
        prefix = self.options_prefix
        m = PETSc.Options(prefix).getInt("sdcit")
        return np.diag(butcher_tableau.c/m)

pcs = ",".join("python" for _ in range(args.stages))

params = {"mat_type": "matfree",
          "snes_type": "ksponly",
          "ksp_type": "gmres",
          "ksp_atol": 1.0e-50,
          "ksp_rtol": 1.0e-5,
          "ksp_converged_rate": None,
          "pc_type": "composite",
          "pc_composite_type": "multiplicative",
          "pc_composite_pcs": pcs,
          "pc_python_type": "__main__.PQPC",
          }

for n in range(args.stages):
    params["sub_"+str(n)]= {
        "pc_python_type": "__main__.PQPC",
        "aux" :
        {
            "sdcit": n+1,
            "pc_type": "fieldsplit",   # block preconditioner
            "pc_fieldsplit_type": "additive",  # block diagonal
            "fieldsplit":  {"ksp_type": "preonly",
                            "pc_type": "lu"}
        }
    }

stepper = TimeStepper(F, butcher_tableau, t, dt, u, bcs=bc,
                      solver_parameters=params)
    
while (float(t) < 1.0):
    if (float(t) + float(dt) > 1.0):
        dt.assign(1.0 - float(t))
    stepper.advance()
    t.assign(float(t) + float(dt))
steps, nits, its = stepper.solver_stats()
error = norm(u-uexact)/norm(uexact)
print(its/steps, error, args.stages)

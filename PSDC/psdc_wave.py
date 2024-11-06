from firedrake import *
from irksome import RadauIIA, Dt, MeshConstant, TimeStepper
from ufl.algorithms.ad import expand_derivatives
from irksome.pc import RanaBase
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='IRK Wave equation solver using parallel SDC preconditioner.')
parser.add_argument('--stages', type=int, default=2, help='Number of stages. Default 2.')

args = parser.parse_known_args()
args = args[0]

butcher_tableau = RadauIIA(args.stages)
N = 50

x0 = 0.0
x1 = 10.0
y0 = 0.0
y1 = 10.0

msh = RectangleMesh(N, N, x1, y1)

MC = MeshConstant(msh)
dt = MC.Constant(10. / N)
t = MC.Constant(0.0)

V = FunctionSpace(msh, "BDM", 2)
Q = FunctionSpace(msh, "DG", 1)
W = V * Q

x, y = SpatialCoordinate(msh)

U = Function(W)
u, p = U.subfunctions
p.interpolate(exp(-(x-x1)**2/2-(y-y1)**2/2))

u0, p0 = split(U)
v, w = TestFunctions(W)
F = inner(Dt(u0), v)*dx + inner(div(u0), w) * dx + inner(Dt(p0), w)*dx - inner(p0, div(v)) * dx

bc = DirichletBC(W.sub(0), as_vector([0, 0]), "on_boundary")

class PQPC(RanaBase):
    def getAtilde(self, A):
        return np.diag(butcher_tableau.c)

params = {"mat_type": "matfree",
          "snes_type": "ksponly",
          "ksp_type": "gmres",
          "ksp_atol": 1.0e-50,
          "ksp_rtol": 1.0e-6,
          "pc_type": "python",
          "pc_python_type": "__main__.PQPC",
          "aux" : 
          {
              "ksp_type": "preonly",
              "pc_type": "lu",
          }
          }

stepper = TimeStepper(F, butcher_tableau, t, dt, U, bcs=bc,
                      solver_parameters=params)
E = 0.5 * (inner(u0, u0)*dx + inner(p0, p0)*dx)
e0 = assemble(E)
while (float(t) < 1.0):
    if float(t) + float(dt) > 1.0:
        dt.assign(1.0 - float(t))

    stepper.advance()

    t.assign(float(t) + float(dt))
steps, nits, its = stepper.solver_stats()
eerror = (e0-assemble(E))/e0
print(its/steps, eerror, args.stages)



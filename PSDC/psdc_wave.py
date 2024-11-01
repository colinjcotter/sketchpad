from firedrake import *
from irksome import RadauIIA, Dt, MeshConstant, TimeStepper
from ufl.algorithms.ad import expand_derivatives
from irksome.pc import RanaBase
import numpy as np

butcher_tableau = RadauIIA(9)
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

bc = DirichletBC(W.sub(0), 0, "on_boundary")

class PQPC(RanaBase):
    def getAtilde(self, A):
        return np.diag(butcher_tableau.c)

params = {"mat_type": "matfree",
          "snes_type": "ksponly",
          "ksp_type": "gmres",
          "ksp_monitor": None,
          "pc_type": "python",
          "pc_python_type": "__main__.PQPC",
          "aux" : 
          {
              "ksp_type": "preonly",
              "pc_type": "lu",
          }
          }

params = {
    "ksp_type": "preonly",
    "pc_type": "lu",
}

params = {"mat_type": "aij",
          "snes_type": "ksponly",
          "ksp_type": "preonly",
          "pc_type": "lu"}

stepper = TimeStepper(F, butcher_tableau, t, dt, U,
                      solver_parameters=params)
#stepper = TimeStepper(F, butcher_tableau, t, dt, U, bcs=bc,
#                      solver_parameters=params)
    
stepper.advance()

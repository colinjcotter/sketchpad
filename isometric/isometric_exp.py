from firedrake import *

n = 50
mesh = UnitSquareMesh(50, 50)

V = VectorFunctionSpace(mesh, "CG", 2, dim=3)
Q = TensorFunctionSpace(mesh, "DG", 1, shape=(3, 2))

W = V*Q*Q #  y, q, p (Lagrange multiplier)

w = Function(W)

y, q, p = split(w)
dy, dq, dp = TestFunctions(W)
n = FacetNormal(mesh)
sigma = Constant(50.)

def d2dn2(r):
    return dot(dot(grad(grad(r)), n), n)

# edge width parameter (from Houston et al)
h = avg(CellVolume(mesh))/FacetArea(mesh)

#

# The biharmonic part
eqn = inner(grad(grad(y)), grad(grad(dy)))*dx
# Consistency term on interior facets
eqn += inner(avg(d2dn2(y)),jump(grad(dy), n))*dS
# Symmetry term on interior facets
eqn += inner(avg(d2dn2(dy)),jump(grad(y), n))*dS
# Penalty term on interior facets
eqn += sigma/h*inner(jump(grad(y), n), jump(grad(dy), n))*dS
# Consistency term on 

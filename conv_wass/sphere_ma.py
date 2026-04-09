from firedrake import *

mesh = IcosahedralSphereMesh(radius=1, refinement_level=4)
deg = 2
V = FunctionSpace(mesh, "CG", deg)

x = SpatialCoordinate(mesh)

def mabs(x):
    return dot(x, x)**0.5

problem = 2

mu0 = Function(V, name="mu0").assign(1.0)

if problem == 1:
    gamma = Constant(1/2)**4 # 1/2 1/4, 1/8, 1/16
    beta = Constant(pi/6)
    alpha = Constant(pi/20)
    xc = as_vector([0., 3**0.5/2, 1/2])

    mu_exp = ((1-gamma)*0.5*(tanh((beta-mabs(x-xc))/alpha)+1)
              + gamma)**0.5
else:
    alpha = Constant(50)
    beta = Constant(5)
    x1 = as_vector([sqrt(3)/2, 0, 0.5])
    x2 = as_vector([-sqrt(3)/2, 0, 0.5])
    def sech(x):
        return 1/cosh(x)

    mu_exp = 1 + alpha*(sech(beta*(inner(x-x1,x-x1)-(pi/2)**2)))**2 \
        + alpha*(sech(beta*(inner(x-x2,x-x2)-(pi/2)**2)))**2

mu1 = Function(V, name="mu1").interpolate(mu_exp)
VTKFile("mu.pvd").write(mu0, mu1)

mu0.assign(mu0/assemble(mu0*dx))
mu1.assign(mu1/assemble(mu1*dx))

eps = Constant(0.01)
gam0 = eps**0.5

u = TrialFunction(V)
du = TestFunction(V)
w = Function(V, name="w").assign(1.0)
v = Function(V, name="v").assign(1.0)
u0 = Function(V)
u1 = Function(V)

nsteps = 10
dt = Constant(0.5*gam0/nsteps)

a = (du*u + dt*inner(grad(du), grad(u)))*dx
Lw = du*u0*dx

Ht_problem = LinearVariationalProblem(a, Lw, u1)
Ht_solver = LinearVariationalSolver(Ht_problem)

res = 10000
tol = 2.e-2
while res > tol:
    u0.assign(w)
    for step in range(nsteps):
        Ht_solver.solve()
        u0.assign(u1)
    resv = norm(v-mu0/u0)
    v.interpolate(mu0/u0)
    u0.assign(v)
    for step in range(nsteps):
        Ht_solver.solve()
        u0.assign(u1)
    resw = norm(w-mu1/u0)
    w.interpolate(mu1/u0)
    res = max(resv, resw)
    print(res)

# moving the mesh
phi = Function(V, name="phi").interpolate(-gam0*ln(v))
W = VectorFunctionSpace(mesh, "CG", 1)
X = Function(W, name="New Coords")

# Rodriguez formula
Rod = cos(mabs(grad(phi)))*x + sin(mabs(grad(phi)))*grad(phi)/mabs(grad(phi))
X.project(Rod)

VTKFile("mu.pvd").write(mu0, mu1, v, w, phi, X)

cellA = Function(V).project(CellVolume(mesh))

new_mesh = Mesh(Function(X))

mu0_mapped = Function(
    functionspaceimpl.WithGeometry.create(mu0.function_space(), new_mesh),
    val=mu0.topological, name="mu0 mapped")
mu1_mapped = Function(
    functionspaceimpl.WithGeometry.create(mu1.function_space(), new_mesh),
    val=mu1.topological, name="mu1 mapped")

V_new = FunctionSpace(new_mesh, "CG", deg)
cellA_mapped = Function(
    functionspaceimpl.WithGeometry.create(cellA.function_space(), new_mesh),
    val=cellA.topological, name="cellA mapped") 
cellB = Function(V_new).project(CellVolume(new_mesh))
scale = Function(V_new).interpolate(mu0_mapped*cellA_mapped/cellB)

VTKFile("mapped.pvd").write(mu0_mapped, mu1_mapped, scale)

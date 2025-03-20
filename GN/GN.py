from firedrake import *

L = 10.
ncells = 10
mesh = PeriodicIntervalMesh(ncells, L)

deg = 1
V = FunctionSpace(mesh, "CG", deg+1)
Q = FunctionSpace(mesh, "DG", deg)
W = V * Q

# lGN = D|u|^2/2 + D^3*(u')^2/6 + D^2*u'*b_t/2 + D*b_t^2/2 - rhobar*g*(D-2*b)

b = Function(Q)
b_t = Function(Q)

g = Constant(10)
rhobar = Constant(1.0)

def variations(U):
    u, D = split(U)
    ell = (
        D*u*u/2 + D**3*u.dx(0)**2/6 + D**2*u.dx(0)*b_t/2
        + D*b_t**2/2 - rhobar*g*D*(D-2*b)
        )*dx
    dl_dU = derivative(ell, U)
    return dl_dU

Un = Function(W)
Unp1 = Function(W)

un, Dn = split(Un)
unp1, Dnp1 = split(Unp1)
uh = (un + unp1)/2
Dh = (Dn + Dnp1)/2

dt = 0.01
dT = Constant(dt)

dl_dU = variations(Un)
du, dD = TestFunctions(W)

dl_dU_h = replace(dl_dU, {un: (un+unp1)/2,
                          Dn: (Dn+Dnp1)/2})

eqn = (unp1 - un)*du*dx + (Dnp1 - Dn)*dD*dx
eqn += replace(dT*dl_dU_h, {du: uh*du.dx(0) - du*uh.dx(0),
                            dD: (du*Dh).dx(0)})

problem = NonlinearVariationalProblem(eqn, Unp1)
solver = NonlinearVariationalSolver(problem)

un_plot, Dn_plot = Un.subfunctions

file0 = VTKFile("GN.pvd")
file0.write(un_plot, Dn_plot)
t = 0.
dumpn = 0
ndump = 10
Tmax = 1.

while t < Tmax - dt/2:
    t += dt
    solver.solve()
    Un.assign(Unp1)

    dumpn += 1
    if dumpn == ndump:
        file0.write(un_plot, Dn_plot)
        dumpn = 0
    

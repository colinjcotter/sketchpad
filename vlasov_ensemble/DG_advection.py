from firedrake import *
import math

ncells = 40
L = 1
mesh = PeriodicIntervalMesh(ncells, L)

V = FunctionSpace(mesh, "DG", 1)
W = VectorFunctionSpace(mesh, "CG", 1)

x, = SpatialCoordinate(mesh)

u = Function(W)

q = Function(V).interpolate(exp(-(x-0.5)**2/(0.2**2/2)))
q_init = Function(V).assign(q)

T = 3
dt = T/500
dtc = Constant(dt)
q_in = Constant(1.0)
m = Constant(1.0)

dq_trial = TrialFunction(V)
phi = TestFunction(V)
a = phi*dq_trial*dx

us = Function(W)
n = FacetNormal(mesh)
un = 0.5*(dot(us, n) + abs(dot(us, n)))

L1 = dtc*(inner(us, grad(phi))*q*dx
          - (phi('+') - phi('-'))*(un('+')*q('+') - un('-')*q('-'))*dS)

q1 = Function(V); q2 = Function(V)
L2 = replace(L1, {q: q1}); L3 = replace(L1, {q: q2})

dq = Function(V)

params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
prob1 = LinearVariationalProblem(a, L1, dq)
solv1 = LinearVariationalSolver(prob1, solver_parameters=params)
prob2 = LinearVariationalProblem(a, L2, dq)
solv2 = LinearVariationalSolver(prob2, solver_parameters=params)
prob3 = LinearVariationalProblem(a, L3, dq)
solv3 = LinearVariationalSolver(prob3, solver_parameters=params)

# The phi solver
Vcg = FunctionSpace(mesh, "CG", 1)
phi_sol = TrialFunction(Vcg)
dphi = TestFunction(Vcg)
phi = Function(Vcg)

nullspace = VectorSpaceBasis(constant=True)

aphi = inner(grad(phi_sol), grad(dphi))*dx
Paphi = phi_sol*dphi*dx + inner(grad(phi_sol), grad(dphi))*dx
F = q*dphi*dx
phi_problem = LinearVariationalProblem(aphi, F, phi, aP=Paphi)
phi_solver = LinearVariationalSolver(phi_problem, nullspace=nullspace,
                                    solver_parameters={
                                        'ksp_type': 'gmres',
                                        #'ksp_monitor': None,
                                        'ksp_atol': 1.0e-11,
                                        #'ksp_converged_reason':None
                                    })
# The u solver
du_trial = TrialFunction(W)
u_test = TestFunction(W)
a = inner(u_test, du_trial)*dx

L1 = -dtc/m*inner(u_test, grad(phi))*dx
du = Function(W)
u1 = Function(W)
u2 = Function(W)
du_prob = LinearVariationalProblem(a, L1, du)
du_solv = LinearVariationalSolver(du_prob)

t = 0.0
step = 0
output_freq = 20

outfile = VTKFile("advection.pvd")
outfile.write(q, phi, u)

while t < T - 0.5*dt:
    phi_solver.solve()
    us.assign(u)
    solv1.solve()
    du_solv.solve()
    q1.assign(q + dq)
    u1.assign(u + du)

    phi_solver.solve()
    us.assign(u1)
    solv2.solve()
    du_solv.solve()
    q2.assign(0.75*q + 0.25*(q1 + dq))
    u2.assign(0.75*u + 0.25*(u1 + du))
    
    phi_solver.solve()
    us.assign(u2)
    solv3.solve()
    du_solv.solve()
    q.assign((1.0/3.0)*q + (2.0/3.0)*(q2 + dq))
    u.assign((1.0/3.0)*u + (2.0/3.0)*(u2 + du))
    
    step += 1
    t += dt

    if step % output_freq == 0:
        outfile.write(q, phi,u)
        print("t=", t)

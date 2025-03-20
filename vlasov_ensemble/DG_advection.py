from firedrake import *
import math

ncells = 40
L = 1
mesh = PeriodicIntervalMesh(ncells, L)

V = FunctionSpace(mesh, "DG", 1)
W = VectorFunctionSpace(mesh, "CG", 1)

x, = SpatialCoordinate(mesh)

u = Function(W).interpolate(as_vector([0.0]))
u0 = Function(W).interpolate(as_vector([0.0]))

q = Function(V).interpolate(exp(-(x-0.5)**2/(0.2**2/2)))
q_init = Function(V).assign(q)

T = 3
dt = T/5000
dtc = Constant(dt)
q_in = Constant(1.0)
m = Constant(1.0)

dq_trial = TrialFunction(V)
phi = TestFunction(V)
a = phi*dq_trial*dx

n = FacetNormal(mesh)
un = 0.5*(dot(u, n) + abs(dot(u, n)))

L1 = dtc*(q*div(phi*u)*dx
          - (phi('+') - phi('-'))*(un('+')*q('+') - un('-')*q('-'))*dS)

q1 = Function(V); q2 = Function(V)
L2 = replace(L1, {q: q1}); L3 = replace(L1, {q: q2})

dq = Function(V)

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
                                         'ksp_monitor': None,
                                         'ksp_atol': 1.0e-11,
                                         'ksp_converged_reason':None
                                     })

params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
prob1 = LinearVariationalProblem(a, L1, dq)
solv1 = LinearVariationalSolver(prob1, solver_parameters=params)
prob2 = LinearVariationalProblem(a, L2, dq)
solv2 = LinearVariationalSolver(prob2, solver_parameters=params)
prob3 = LinearVariationalProblem(a, L3, dq)
solv3 = LinearVariationalSolver(prob3, solver_parameters=params)

t = 0.0
step = 0
output_freq = 20

outfile = VTKFile("advection.pvd")
outfile.write(q, u)

while t < T - 0.5*dt:
    u0.assign(u)
    phi_solver.solve()
    solv1.solve()
    q1.assign(q + dq)
    u.interpolate(u0 - dtc*grad(phi)/m)
    print( norm(u), norm(phi), norm(q))

    phi_solver.solve()
    solv2.solve()
    q2.assign(0.75*q + 0.25*(q1 + dq))
    u.interpolate(0.75*u0 - dtc*0.25*(u - grad(phi)/m))

    phi_solver.solve()
    solv3.solve()
    q.assign((1.0/3.0)*q + (2.0/3.0)*(q2 + dq))
    u.interpolate((1.0/3.0)*u0 - dtc*(2.0/3.0)*(u - grad(phi)/m))
    
    step += 1
    t += dt

    if step % output_freq == 0:
        outfile.write(q, u)
        print("t=", t)

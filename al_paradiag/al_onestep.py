from firedrake import *
from petsc4py import PETSc
PETSc.Sys.popErrorHandler()

n = 100
mesh = UnitSquareMesh(n, n)

dt = Constant(0.1)
gamma = Constant(1.0e4)

degree = 1
V = FunctionSpace(mesh, "BDM", degree) # velocity space
Q = FunctionSpace(mesh, "DG", degree-1)

W = V * Q

class ALWaveSchurPC(PCBase):
    """A matrix free operator that implements the approximate Schur
    complement for the discrete wave equation``.
    """
    def initialize(self, pc):

        self.xf = Function(Q) # input to apply
        self.yf = Function(Q) # output to apply
        
        # set up Riesz map to a function from self.xf to r
        self.r = Function(Q)
        r_in = Function(Q)
        dparams = {'ksp_type':'preonly',
                   'pc_type':'lu'}
        riesz_projector = Projector(6*r_in, self.r,
                                    solver_parameters=dparams)
        self.riesz_solver = riesz_projector.solver
        
        # set up solver to apply delta to r and put result in self.delta_r
        self.delta_r = Function(V)
        u = TrialFunction(V)
        v = TestFunction(V)
        zvec = as_vector((Constant(0.), Constant(0.)))
        bcs = [DirichletBC(V, zvec, "on_boundary")]
        delta_eqn = (inner(u, v) + div(v)*self.r)*dx
        dparams = {'ksp_type':'preonly',
                   'pc_type':'lu'}
        delta_prob = LinearVariationalProblem(lhs(delta_eqn),
                                              rhs(delta_eqn),
                                              self.delta_r, bcs=bcs)
        self.delta_solver = LinearVariationalSolver(delta_prob,
                                                    solver_parameters=dparams)

        # Then apply divergence via projection from self.delta_r to self.yf
        self.divergence_projector = Projector(div(self.delta_r),
                                              self.yf,
                                              solver_parameters=dparams)
        
    def update(self, pc):
        pass

    def apply(self, pc, X, Y):
        # copy petsc vec into Function
        with self.xf.dat.vec_wo as v:
            X.copy(v)

        # first need to Riesz map to a function from self.xf to r
        self.riesz_solver.solve(self.r, self.xf)

        # Then apply delta to r and put result in self.delta_r
        self.delta_solver.solve()
        # Then apply divergence via projection from self.delta_r to self.yf
        self.divergence_projector.project()
        # rescale with coefficients

        self.yf *= -gamma*dt/2
        
        with self.yf.dat.vec_ro as v:
            v.copy(Y)

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError

W = V * Q
u, p = TrialFunctions(W)
v, q = TestFunctions(W)
x, y = SpatialCoordinate(mesh)

u0 = Function(V).interpolate(as_vector([cos(sin(2*pi*x)+cos(pi*y))+exp(cos(4*pi*y)),x*y]))
p0 = Function(Q).interpolate(exp(sin(pi*x)*sin(pi*y)))
One = Function(Q).assign(1.0)
p0 -= assemble(p0*dx)/assemble(One*dx)
eqn = (inner(v,u - u0) - dt*div(v)*(p + p0)/2 + q*(p - p0)
       + dt*q*div((u+u0)/2))*dx
# the gamma term
eqn += gamma*div(v)*(p - p0 + dt*div((u+u0)/2))*dx

luparams = {
    'mat_type': 'aij',
    'ksp_type': 'gmres',
    'ksp_monitor': None,
    'pc_type':'lu',
    'pc_factor_mat_solver_type':'mumps'
}

schur_params = {
    #'ksp_view': None,
    'ksp_type': 'fgmres',
    'ksp_monitor': None,
    'ksp_monitor_true_residual': None,
    'pc_type':'fieldsplit',
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "lu",
    "fieldsplit_1_ksp_type": "gmres",
    "fieldsplit_1_ksp_max_it": 1,
    "fieldsplit_1_pc_type": "python",
    "fieldsplit_1_pc_python_type": "__main__.ALWaveSchurPC"
}

v_basis = VectorSpaceBasis(constant=True)
nullspace = MixedVectorSpaceBasis(W, [W.sub(0), v_basis])
w1 = Function(W)
zvec = as_vector((Constant(0.), Constant(0.)))
bcs = [DirichletBC(W.sub(0), zvec, "on_boundary")]
up_prob = LinearVariationalProblem(lhs(eqn), rhs(eqn), w1,bcs = bcs)
up_solver = LinearVariationalSolver(up_prob,
                                   nullspace=nullspace,
                                   solver_parameters=schur_params)

up_solver.solve()

u1, p1 = w1.subfunctions
File("alonestep.pvd").write(u1, p1)

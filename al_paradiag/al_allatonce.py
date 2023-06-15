from firedrake import *

n = 20
m = 3
mesh = UnitSquareMesh(n, m)

dt = Constant(0.1)
gamma = Constant(1.0e4)

degree = 1
ntime = 10
V0 = FunctionSpace(mesh, "BDM", degree)
Q0 = FunctionSpace(mesh, "DG", degree-1)
V = VectorFunctionSpace(mesh, "BDM", degree, dim=ntime) # velocity space
Q = VectorFunctionSpace(mesh, "DG", degree-1, dim=ntime)

W = V * Q

class ALWaveSchurPC(PCBase):
    """A matrix free operator that implements the approximate Schur
    complement for the discrete wave equation``.
    """
    def initialize(self, pc):
        prefix = pc.getOptionsPrefix()
        options_prefix = prefix + "ALwave_"
        # we assume P has things stuffed inside of it
        _, P = pc.getOperators()
        context = P.getPythonContext()

        test, trial = context.a.arguments()

        if test.function_space() != trial.function_space():
            raise ValueError("ALWaveSchurPC only makes sense if test and trial space are the same")

        self.xf = Function(Q) # input to apply
        self.yf = Function(Q) # output to apply
        self.zf = Function(Q) # input to all at once mass system
        
        # set up Riesz map to a function from self.xf to r
        self.r = Function(Q)
        r_in = Function(Q)
        dparams = {'ksp_type':'preonly',
                   'pc_type':'lu'}
        self.riesz_projector = Projector(r_in, self.r,
                                         solver_parameters=dparams)

        # set up solver to apply delta to r and put result in self.delta_r
        self.delta_r = Function(V)
        u = TrialFunction(V)
        v = TestFunction(V)
        bcs = [DirichletBC(V, Constant(0.), "on_boundary")]
        delta_eqn = (inner(u[0, :], v[0, :]) + div(v[0, :])*self.r[0])*dx
        for i in range(1, ntime):
            delta_eqn += (inner(u[i, :], v[i, :]) + div(v[i, :])*self.r[i])*dx
        dparams = {'ksp_type':'preonly',
                   'pc_type':'lu'}
        delta_prob = LinearVariationalProblem(lhs(delta_eqn),
                                              rhs(delta_eqn),
                                              self.delta_r, bcs=bcs)
        self.delta_solver = LinearVariationalSolver(delta_prob,
                                                    solver_parameters=dparams)

        # Then apply divergence via projection from self.delta_r to self.zf
        self.divergence_projector = Projector(div(self.delta_r),
                                              self.zf,
                                              solver_parameters=dparams)

    def update(self, pc):
        pass

    def apply(self, pc, X, Y):
        # copy petsc vec into Function
        with self.xf.dat.vec_wo as v:
            X.copy(v)

        # first need to Riesz map to a function from self.xf to r
        self.r.assign(self.riesz_projector.applyinv(self.xf))
        # Then apply delta to r and put result in self.delta_r
        self.delta_solver.solve()
        # Then apply divergence via projection from self.delta_r to self.yf
        self.divergence_projector.project()
        # rescale with coefficients
        print("Applying")
        self.zf *= -gamma*dt/2

        # finally, solve the all at once system with zf using back substitution
        self.yf[0].assign(self.zf[0])
        for i in range(1, ntime):
            self.yf[i].assign(self.yf[i-1] + self.zf[i])
        
        with self.yf.dat.vec_ro as v:
            v.copy(Y)

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError

W = V * Q
u, p = TrialFunctions(W)
v, q = TestFunctions(W)
x, y = SpatialCoordinate(mesh)

u0 = Function(V0).interpolate(as_vector([cos(sin(2*pi*x)+cos(pi*y))+exp(cos(4*pi*y)),x*y]))
p0 = Function(Q0).interpolate(exp(sin(pi*x)*sin(pi*y)))
One = Function(Q0).assign(1.0)
p0 -= assemble(p0*dx)/assemble(One*dx)

for i in range(ntime):
    vp = v[i, :]
    qp = q[i]
    u1 = u[i, :]
    p1 = p[i]
    if i > 0:
        u0 = u[i-1, :]
        p0 = p[i-1]
    eqn0 = (inner(vp, u1 - u0) - dt*div(vp)*(p1 + p0)/2 + qp*(p1 - p0)
       + dt*qp*div((u1+u0)/2))*dx
    # the gamma term
    eqn0 += (gamma*div(vp)*(p1 - p0) + dt*div((u1+u0)/2))*dx

    if i == 0:
        eqn = eqn0
    else:
        eqn += eqn0

luparams = {
    'mat_type': 'aij',
    'ksp_type': 'gmres',
    'ksp_monitor': None,
    'pc_type':'lu',
    'pc_factor_mat_solver_type':'mumps'
}

# Note, we could also solve the 0 block using back substitution

schur_params = {
    'ksp_type': 'gmres',
    'ksp_monitor': None,
    'pc_type':'fieldsplit',
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "pc_fieldsplit_0_ksp_type": "preonly",
    "pc_fieldsplit_0_pc_type": "lu",
    "pc_fieldsplit_1_ksp_type": "preonly",
    "pc_fieldsplit_1_pc_type": "python",
    "pc_fieldsplit_1_pc_python_type": "__main__.ALWaveSchurPC"
}

v_basis = VectorSpaceBasis(constant=True)
nullspace = MixedVectorSpaceBasis(W, [W.sub(0), v_basis])
w1 = Function(W)

vec = [Constant(0.), Constant(0.)]
veclist = []
for i in range(ntime):
    veclist.append(vec)

zvec = as_tensor(veclist)
bcs = [DirichletBC(W.sub(0), zvec, "on_boundary")]
up_prob = LinearVariationalProblem(lhs(eqn), rhs(eqn), w1,bcs = bcs)
up_solver = LinearVariationalSolver(up_prob,
                                   nullspace=nullspace,
                                   solver_parameters=schur_params)

up_solver.solve()

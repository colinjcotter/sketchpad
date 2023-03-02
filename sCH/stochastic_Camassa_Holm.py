from firedrake import *
from firedrake_adjoint import *
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from nudging.model import *
import numpy as np

class Camsholm(base_model):
    def __init__(self, n, nsteps, dt = 0.01, alpha=1.0, seed=12353):

        self.n = n
        self.nsteps = nsteps
        self.alpha = alpha
        self.dt = dt
        self.seed = seed

    def setup(self, comm = MPI.COMM_WORLD):
        self.mesh = PeriodicIntervalMesh(self.n, 40.0, comm = comm) # mesh need to be setup in parallel
        self.x, = SpatialCoordinate(self.mesh)

        #FE spaces
        self.V = FunctionSpace(self.mesh, "CG", 1)
        self.W = MixedFunctionSpace((self.V, self.V))
        self.w0 = Function(self.W)
        self.m0, self.u0 = self.w0.split()       

        #Interpolate the initial condition

        #Solve for the initial condition for m.
        alphasq = self.alpha**2
        self.p = TestFunction(self.V)
        self.m = TrialFunction(self.V)
        
        self.am = self.p*self.m*dx
        self.Lm = (self.p*self.u0 + alphasq*self.p.dx(0)*self.u0.dx(0))*dx
        mprob = LinearVariationalProblem(self.am, self.Lm, self.m0)
        solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'}
        self.msolve = LinearVariationalSolver(mprob,
                                              solver_parameters=solver_parameters)
        
        #Build the weak form of the timestepping algorithm. 

        self.p, self.q = TestFunctions(self.W)

        self.w1 = Function(self.W)
        self.w1.assign(self.w0)
        self.m1, self.u1 = split(self.w1)   # for n+1 the  time
        self.m0, self.u0 = split(self.w0)   # for n th time 
        
        # stochastic term
        self.dW = Function(self.V)

        # elliptic problem to have smoother noise in space
        p = TestFunction(self.V)
        q = TrialFunctino(self.V)
        self.xi = Function(self.V)
        a = inner(grad(p), grad(q))*dx + p*q*dx
        L = p*self.xi*dx
        dW_prob = LinearVariationalProblem(a, L, self.dW)
        self.dw_solver = LinearVariationalSolver(dW_prob,
                                                 solver_parameters={'mat_type': 'aij', 'ksp_type': 'preonly','pc_type': 'lu'})
        
        # finite element linear functional
        Dt = Constant(self.dt)
        self.mh = 0.5*(self.m1 + self.m0)
        self.uh = 0.5*(self.u1 + self.u0)
        self.v = self.uh*Dt+self.dW*Dt**0.5

        self.L = ((self.q*self.u1 + alphasq*self.q.dx(0)*self.u1.dx(0) - self.q*self.m1)*dx +(self.p*(self.m1-self.m0) + (self.p*self.v.dx(0)*self.mh -self.p.dx(0)*self.v*self.mh))*dx)

        #def Linearfunc

        # solver

        self.uprob = NonlinearVariationalProblem(self.L, self.w1)
        self.usolver = NonlinearVariationalSolver(self.uprob, solver_parameters={'mat_type': 'aij', 'ksp_type': 'preonly','pc_type': 'lu'})

        # Data save
        self.m0, self.u0 = self.w0.split()
        self.m1, self.u1 = self.w1.split()

        # state for controls
        self.X = self.allocate()

        # vertex only mesh for observations
        x_obs = np.arange(0.5,40.0)
        x_obs_list = []
        for i in x_obs:
            x_obs_list.append([i])
        self.VOM = VertexOnlyMesh(self.mesh, x_obs_list)
        self.VVOM = FunctionSpace(self.VOM, "DG", 0)

    def run(self, X0, X1):
        for i in range(len(X0)):
            self.X[i].assign(X0[i])
        self.w0.assign(self.X[0])
        self.msolve.solve()
        for step in range(self.nsteps):
            self.xi.assign(self.X[step+1])
            self.dW_solver.solve()
            self.usolver.solve()
            self.w0.assign(self.w1)
        X1[0].assign(self.w0) # save sol at the nstep th time 

    def controls(self):
        controls_list = []
        for i in range(len(self.X)):
            controls_list.append(Control(self.X[i]))
        return controls_list
        
    def obs(self):
        m, u = self.w0.split()
        Y = Function(self.VVOM)
        Y.interpolate(u)
        return Y

    def allocate(self):
        particle = [Function(self.W)]
        for i in range(self.nsteps):
            dW = Function(self.V)
            particle.append(dW)
        return particle

    def randomize(self, X, c1=0, c2=1, gscale=None, g=None):
        rg = self.rg
        count = 0
        for i in range(self.nsteps):
            count += 1
            X[count].assign(c1*X[count] + c2*rg.normal(self.V, 0., 1.0))
            if g:
                X[count] += gscale*g[count]

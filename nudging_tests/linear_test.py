from firedrake import *
from pyop2.mpi import MPI
from nudging import *
import numpy as np

# model
# multiply by A and add D
T = 1.
nsteps = 10
dt = T/nsteps
model = LGModel(A=1., D=2., nsteps=nsteps, dt=dt)

# solving
# dx = A*x*dt + D*dW
# integrating factor e^{-At}
# e^{-At}dx = e^{-At}*x*dt + De^{-t}dW
# we have d(e^{-At}x) = e^{-At}dx - e^{-At}x*dt
# so
# d(e^{-At}x) = De^{-At}dW
# i.e.
# x(t) = e^{At}x(0) + int_0^t D e^{A(t-s)}dW(s)

# E(int_0^t D e^{A(t-s)}dW(s)) = 0
# E[(int_0^t D e^{A(t-s)}dW(s))^2] = int_0^t D^2 e^{2A(t-s)} ds (Ito isometry)
# = (-D^2/(2A))*[e^{2A(t-s)}]_{s=0}^{s=t}
# = (D^2/(2A))*(e^{2At} - 1)
# so pi(x(t)|x(0)) ~ N(e^{at}x(0), sig^2)
# sig^2 = (D^2/2A)*(e^{2At} - 1)

# take x(0) ~ N(c, d^2)

# write y(t) = x(t) - e^{At}x(0),
# then
# z = (y(t), x(0))^T
# and z ~ N( (0, c)^T, Sigma )
# with
# Sigma = (sig^2   0)
#         (0     d^2)

# (x(t), x(0))^T = Bz with
# B = (1  exp(At))
#     (0        1)

# (x(t), x(0))^T ~ N(B(0, c)^T, B Sigma B^T)

# Sigma B^T = (sig^2   0)(1       0) = (sig^2         0)
#             (0     d^2)(exp(At) 1)   (d^2*exp(At) d^2)

# B Sigma B^T = (1 exp(At))(sig^2         0)
#               (0       1)(d^2*exp(At) d^2)

#             = (sig^2 + d^2*exp(2At) d^2*exp(At))
#               (d^2*exp(At)                  d^2)

# B(0, c)^T = (1 exp(At))(0) = c(exp(At))
#             (0       1)(c)    (      1)

# The marginal distribution for x(t) is then
# x(t) ~ N(c*exp(At), sig^2 + d^2*exp(2At))

# now we have an observation y = x(1) + e, e ~ N(0, S^2)
# Reverend Bayes
# pi(x(1)|y) propto pi(y|x(1))*pi(x(1))
# = exp( -(1/2)[(y-x(1))^2/S^2 + (x(1)-c*exp(A))^2/(sig^2+d^2*exp(2A))])
# = exp( -(1/2)[(y-x(1))^2/S^2 + (x(1)-a)^2/b^2])
# where a = c*exp(A), b = (sig^2+d^2*exp(2A))
# completing the square
# (y-x(1))^2/S^2 + (x(1)-a)^2/b^2
# = (x(1)^2-2x(1)y + y^2)/S^2 + (x(1)^2 - 2ax(1) + a^2)/b^2
# = x(1)**2(b^2 + S^2)/(b^2S^2) - 2x(1)(b^2y + S^2a)/(b^2S^2) + stuff
# = (b^2 + S^2)/(b^2S^2)[ x(1)^2 - 2x(1)*(b^2y + S^2a)/(b^2+S^2)] + stuff
# = (b^2 + S^2)/(b^2S^2)[ x(1)^2 - 2x(1)*(b^2y + S^2a)/(b^2+S^2)] + stuff
# = (b^2 + S^2)/(b^2S^2)[ x(1)^2 - (b^2y + S^2a)/(b^2+S^2)]^2 + stuff
# where stuff is anything independent of x(1)

# then
# x(1)|y ~ N((b^2y + S^2a)/(b^2+S^2), (b^2S^2)/(b^2 + S^2))

# bootstrap filter
bsfilter = bootstrap_filter()

nensemble = [20]*5
bsfilter.setup(nensemble, model)

# data
y = model.obs()
y0 = 1.2
y.dat.data[:] = y0

# prepare the initial ensemble
a = 1.
b = 1.
for i in range(nensemble[bsfilter.ensemble_rank]):
    dx0 = model.rg.normal(model.R, a, b)
    u = bsfilter.ensemble[i][0]
    u.assign(dx0)

# observation noise standard deviation
S = 0.05
    
def log_likelihood(y, Y):
    ll = (y-Y)**2/S**2/2*dx
    return ll

bsfilter.assimilation_step(y, log_likelihood)

#jtfilter = jittertemp_filter(n_jitt = 10, delta = 0.01,
#                             verbose=verbose, MALA=MALA,
#                             nudging=nudging,
#                             visualise_tape=visualise_tape)

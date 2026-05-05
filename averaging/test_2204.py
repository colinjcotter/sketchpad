from firedrake import *
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
import asQ

# set arguments
ref_level = 3
hours = 0.5
dt = 60*60*hours
alpha = 1
ns = 4
nt = 4
theta = 0.5
nslices = 2

### === --- Experimental setup for Williamson 5  --- === ###
# setup slice length and check if they are integer
slice_length_t = int(nt/nslices)
assert(slice_length_t*nslices == nt)

# setup time_partition # I am assuming nt and ns are the same for now
time_partition_t = [slice_length_t for _ in range(nslices)]

# setup ensemble
ensemble = asQ.create_ensemble(time_partition_t)
ensemble_rank = ensemble.ensemble_comm.rank

# set mesh and parameters for Williamsom 5 test
R0 = 6371220.
mesh_degree = 1

# create mesh
mesh = IcosahedralSphereMesh(radius=R0, refinement_level=ref_level,
                             degree=mesh_degree, name="mesh", comm=ensemble.comm)
cx = SpatialCoordinate(mesh)
mesh.init_cell_orientations(cx)
PETSc.Sys.Print("Created mesh")
    
V1 = FunctionSpace(mesh, "BDM", 2)
V2 = FunctionSpace(mesh, "DG", 1)
W = MixedFunctionSpace((V1, V2))

### === --- Set up AllAtOnceFunctions --- === ###
Wall = asQ.AllAtOnceFunction(ensemble, time_partition_t, W)

### === --- Set up form_function and form_mass for ParaDiag --- === ###
def form_mass(uu, up, vu, vp):
    return (inner(uu, vu) + up * vp) * dx

def form_mass_t(uu, up, vu, vp, t):
    return (inner(uu, vu) + up * vp) * dx


### === --- Set up AllAtOnceSolver for forward nt propagation --- === ###
#propagate_form = asQ.AllAtOnceForm(Wall, dt/nt, theta,
#                                   form_mass, get_form_function())
propagate_form = asQ.AllAtOnceForm(Wall, dt/nt, theta,
                                   form_mass, form_mass_t)
propagate_solver = asQ.AllAtOnceSolver(propagate_form, Wall,
                                       options_prefix="propagate_solver")

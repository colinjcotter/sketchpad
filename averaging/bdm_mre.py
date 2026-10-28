from firedrake import *

# Parameters
R0 = 6371220.0
Omega = Constant(7.292e-5)
g = Constant(9.8)
u_max = Constant(20.0)

# Mesh
degree = 3
mesh = IcosahedralSphereMesh(
    radius=R0,
    refinement_level=3,
    degree=degree,
    name="mesh",
)
mesh.init_cell_orientations(SpatialCoordinate(mesh))

x, y, z = SpatialCoordinate(mesh)

# Function spaces
V1 = FunctionSpace(mesh, "BDM", 2)
V2 = FunctionSpace(mesh, "DG", 1)

# Initial conditions
u_expr = as_vector([-u_max*y/R0, u_max*x/R0, 0.0])
eta_expr = - ((R0 * Omega * u_max + u_max*u_max/2.0)*(z*z/(R0*R0)))/g
un = Function(V1, name="Velocity").project(u_expr)
etan = Function(V2, name="Elevation").interpolate(eta_expr)

# Original integrals
velocity_original = assemble(dot(un, un) * dx)
elevation_original = assemble(etan * etan * dx)

# Save checkpoint
filename = "bdm_mre.h5"

with CheckpointFile(filename, "w") as checkpoint:
    checkpoint.save_mesh(mesh)
    checkpoint.save_function(un)
    checkpoint.save_function(etan)

# Load checkpoint
with CheckpointFile(filename, "r") as checkpoint:
    mesh_loaded = checkpoint.load_mesh("mesh")
    un_loaded = checkpoint.load_function(mesh_loaded, "Velocity")
    etan_loaded = checkpoint.load_function(mesh_loaded, "Elevation")

# Loaded integrals
velocity_loaded = assemble(dot(un_loaded, un_loaded) * dx)
elevation_loaded = assemble(etan_loaded * etan_loaded * dx)

# Compare
def compare(label, original, loaded):
    absolute_difference = abs(original - loaded)
    relative_difference = absolute_difference / abs(original)

    print(f"\n{label}")
    print("Original integral =", original)
    print("Reloaded integral =", loaded)
    print("Absolute difference =", absolute_difference)
    print("Relative difference =", relative_difference)


print("degree =", degree)

compare("BDM2 Velocity", velocity_original, velocity_loaded)
compare("DG1 Elevation", elevation_original, elevation_loaded)


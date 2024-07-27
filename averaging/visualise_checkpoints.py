from firedrake import *
from firedrake.__future__ import interpolate
import numpy as np
import argparse
print = PETSc.Sys.Print

def get_latlon_mesh(mesh):
    coords_orig = mesh.coordinates
    coords_fs = coords_orig.function_space()

    if coords_fs.extruded:
        cell = mesh._base_mesh.ufl_cell().cellname()
        DG1_hori_elt = FiniteElement("DG", cell, 1, variant="equispaced")
        DG1_vert_elt = FiniteElement("DG", interval, 1, variant="equispaced")
        DG1_elt = TensorProductElement(DG1_hori_elt, DG1_vert_elt)
    else:
        cell = mesh.ufl_cell().cellname()
        DG1_elt = FiniteElement("DG", cell, 1, variant="equispaced")
    vec_DG1 = VectorFunctionSpace(mesh, DG1_elt)
    coords_dg = Function(vec_DG1).interpolate(coords_orig)
    coords_latlon = Function(vec_DG1)
    shapes = {"nDOFs": vec_DG1.finat_element.space_dimension(), 'dim': 3}

    radius = np.min(np.sqrt(coords_dg.dat.data[:, 0]**2 + coords_dg.dat.data[:, 1]**2 + coords_dg.dat.data[:, 2]**2))
    # lat-lon 'x' = atan2(y, x)
    coords_latlon.dat.data[:, 0] = np.arctan2(coords_dg.dat.data[:, 1], coords_dg.dat.data[:, 0])
    # lat-lon 'y' = asin(z/sqrt(x^2 + y^2 + z^2))
    coords_latlon.dat.data[:, 1] = np.arcsin(coords_dg.dat.data[:, 2]/np.sqrt(coords_dg.dat.data[:, 0]**2 + coords_dg.dat.data[:, 1]**2 + coords_dg.dat.data[:, 2]**2))
    # our vertical coordinate is radius - the minimum radius
    coords_latlon.dat.data[:, 2] = np.sqrt(coords_dg.dat.data[:, 0]**2 + coords_dg.dat.data[:, 1]**2 + coords_dg.dat.data[:, 2]**2) - radius

# We need to ensure that all points in a cell are on the same side of the branch cut in longitude coords
# This kernel amends the longitude coords so that all longitudes in one cell are close together
    kernel = op2.Kernel("""
#define PI 3.141592653589793
#define TWO_PI 6.283185307179586
void splat_coords(double *coords) {{
    double max_diff = 0.0;
    double diff = 0.0;

    for (int i=0; i<{nDOFs}; i++) {{
        for (int j=0; j<{nDOFs}; j++) {{
            diff = coords[i*{dim}] - coords[j*{dim}];
            if (fabs(diff) > max_diff) {{
                max_diff = diff;
            }}
        }}
    }}

    if (max_diff > PI) {{
        for (int i=0; i<{nDOFs}; i++) {{
            if (coords[i*{dim}] < 0) {{
                coords[i*{dim}] += TWO_PI;
            }}
        }}
    }}
}}
""".format(**shapes), "splat_coords")

    op2.par_loop(kernel, coords_latlon.cell_set,
                 coords_latlon.dat(op2.RW, coords_latlon.cell_node_map()))
    return Mesh(coords_latlon)

#get command arguments
parser = argparse.ArgumentParser(description='Visualising data in a .h5 file.')
parser.add_argument('--filename', type=str, default='filename')
args = parser.parse_known_args()
args = args[0]
print(args)

name = args.filename

print('Visualise data in ' + name + ".h5")

print("\n")
print("=== content in " + name + " ===")
with CheckpointFile(name+".h5", 'r') as checkpoint:
    mesh = checkpoint.load_mesh("mesh")
    timestepping_history = checkpoint.get_timestepping_history(mesh, name="Velocity")
    length = len(timestepping_history["time"])
    print("timestepping_history = ", timestepping_history)
    print("timestepping_history_index = ", timestepping_history["index"])
    print("timestepping_history_time = ", timestepping_history["time"])
    print("timestepping_history_tdump = ", timestepping_history["tdump"])
    print("timestepping_history_tcheck = ", timestepping_history["tcheck"])
    print("timestepping_history_index_last = ", timestepping_history["index"][-1])
    print("timestepping_history_time_last = ", timestepping_history["time"][-1])
    print("timestepping_history_tdump_last = ", timestepping_history["tdump"][-1])
    print("timestepping_history_tcheck_last = ", timestepping_history["tcheck"][-1])


print("\n")
print("=== visualise data in " + name + ".h5 on a latlon grid ===")
mesh_ll = get_latlon_mesh(mesh)
global_normal_ll = as_vector([0, 0, 1])
mesh_ll.init_cell_orientations(global_normal_ll)
outfile = output.VTKFile(name+'_latlon.pvd', mode="a")

for i in range(length):
    print("Visualising the result at t = ", timestepping_history["time"][i])
        
    with CheckpointFile(name+".h5", 'r') as checkpoint:
        un = checkpoint.load_function(mesh, "Velocity", idx=i)
        etan = checkpoint.load_function(mesh, "Elevation", idx=i)
        PV = checkpoint.load_function(mesh, "PotentialVorticity", idx=i)
        b = checkpoint.load_function(mesh, "Topography")
        
        field_un = Function(
            functionspaceimpl.WithGeometry.create(un.function_space(), mesh_ll),
            val=un.topological)
        field_etan = Function(
            functionspaceimpl.WithGeometry.create(etan.function_space(), mesh_ll),
            val=etan.topological)
        field_PV = Function(
            functionspaceimpl.WithGeometry.create(PV.function_space(), mesh_ll),
            val=PV.topological)
        field_b = Function(
            functionspaceimpl.WithGeometry.create(b.function_space(), mesh_ll),
            val=b.topological)

    outfile.write(field_un, field_etan, field_PV, field_b)

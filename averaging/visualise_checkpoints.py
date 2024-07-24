from firedrake import *
from latlon import *
from firedrake.__future__ import interpolate
import numpy as np
import argparse
print = PETSc.Sys.Print

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

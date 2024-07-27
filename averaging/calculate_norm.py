from firedrake import *
from firedrake.__future__ import interpolate
import numpy as np
import argparse
print = PETSc.Sys.Print

#get command arguments
parser = argparse.ArgumentParser(description='Comparing data in .h5 files and calculate norms.')
parser.add_argument('--file0', type=str, default='file0')
parser.add_argument('--file1', type=str, default='file1')
parser.add_argument('--meshdir', type=str, default='.')
args = parser.parse_known_args()
args = args[0]
print(args)

file0 = args.file0
file1 = args.file1
meshdir = args.meshdir

print('calculate normalised norms in ' + file0 + " with respect to " + file1)

print("\n")
print("=== pickup mesh ===")
#pickup mesh
with CheckpointFile(meshdir+"/mesh.h5", 'r') as checkpoint:
    mesh = checkpoint.load_mesh("mesh")
    x = SpatialCoordinate(mesh)
    print("Picked up the mesh from mesh.h5")

print("\n")
print("=== content in " + file0 + " ===")
with CheckpointFile(file0+".h5", 'r') as checkpoint0:
    mesh0 = checkpoint0.load_mesh("mesh")
    timestepping_history0 = checkpoint0.get_timestepping_history(mesh0, name="Velocity")
    length0 = len(timestepping_history0["time"])
    print("timestepping_history = ", timestepping_history0)
    print("timestepping_history_index = ", timestepping_history0["index"])
    print("timestepping_history_time = ", timestepping_history0["time"])
    print("timestepping_history_tdump = ", timestepping_history0["tdump"])
    print("timestepping_history_tcheck = ", timestepping_history0["tcheck"])
    print("timestepping_history_index_last = ", timestepping_history0["index"][-1])
    print("timestepping_history_time_last = ", timestepping_history0["time"][-1])
    print("timestepping_history_tdump_last = ", timestepping_history0["tdump"][-1])
    print("timestepping_history_tcheck_last = ", timestepping_history0["tcheck"][-1])

    un = checkpoint0.load_function(mesh, "Velocity", idx=0)
    testc0 = assemble(dot(un,un)*dx)
    print("testc0 = ", testc0)

print("\n")
print("=== content in " + file1 + " ===")
with CheckpointFile(file1+".h5", 'r') as checkpoint1:
    mesh1 = checkpoint1.load_mesh("mesh")
    timestepping_history1 = checkpoint1.get_timestepping_history(mesh1, name="Velocity")
    length1 = len(timestepping_history1["time"])
    print("timestepping_history = ", timestepping_history1)
    print("timestepping_history_index = ", timestepping_history1["index"])
    print("timestepping_history_time = ", timestepping_history1["time"])
    print("timestepping_history_tdump = ", timestepping_history1["tdump"])
    print("timestepping_history_tcheck = ", timestepping_history1["tcheck"])
    print("timestepping_history_index_last = ", timestepping_history1["index"][-1])
    print("timestepping_history_time_last = ", timestepping_history1["time"][-1])
    print("timestepping_history_tdump_last = ", timestepping_history1["tdump"][-1])
    print("timestepping_history_tcheck_last = ", timestepping_history1["tcheck"][-1])

    un = checkpoint1.load_function(mesh, "Velocity", idx=0)
    testc1 = assemble(dot(un,un)*dx)
    print("testc1 = ", testc1)
    
print("\n")
print("=== calculate normalised norms in " + file0 + " with respect to " + file1 + " ===")
etanorm_list = []
unorm_list = []
for i in range(min(length0, length1)):
    assert(timestepping_history0["time"][i] == timestepping_history1["time"][i])
    print("Picking up the results at t = ", timestepping_history0["time"][i])
    #calculate norms
    with CheckpointFile(file0+".h5", 'r') as checkpoint0:
        un0 = checkpoint0.load_function(mesh, "Velocity", idx=i)
        etan0 = checkpoint0.load_function(mesh, "Elevation", idx=i)
    with CheckpointFile(file1+".h5", 'r') as checkpoint1:
        un1 = checkpoint1.load_function(mesh, "Velocity", idx=i)
        etan1 = checkpoint1.load_function(mesh, "Elevation", idx=i)

    etanorm = errornorm(etan0, etan1)/norm(etan1)
    unorm = errornorm(un0, un1, norm_type="Hdiv")/norm(un1, norm_type="Hdiv")
    print('etanorm', etanorm, 'unorm', unorm)
    etanorm_list.append(Constant(etanorm))
    unorm_list.append(Constant(unorm))

print("etanorm_list = ", etanorm_list)
print("unorm_list = ", unorm_list)

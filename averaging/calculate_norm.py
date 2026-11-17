from firedrake import *
from firedrake.__future__ import interpolate
import numpy as np
import argparse
print = PETSc.Sys.Print

#get command arguments
parser = argparse.ArgumentParser(description='Comparing data in .h5 files and calculate norms.')
parser.add_argument('--file0', type=str, default='file0')
parser.add_argument('--file1', type=str, default='file1')
args = parser.parse_known_args()
args = args[0]
print(args)

#set arguments
file0 = args.file0
file1 = args.file1

# print out the content of the checkpoint file
def view_checkpoint(filename):
    print("\n")
    print("=== content in " + filename + ".h5 ===")
    #open checkpoint and pickup results at the last index
    with CheckpointFile(filename+".h5", 'r') as checkpoint:
        mesh = checkpoint.load_mesh("mesh")
        uini = checkpoint.load_function(mesh, "Velocity", idx=0)
        etaini = checkpoint.load_function(mesh, "Elevation", idx=0)
        X3 = checkpoint.load_function(mesh, "coordinates")
        history = checkpoint.get_timestepping_history(mesh, name="Velocity")
        length = len(history["time"])
        last_index = int(history["index"][-1])
        last_time = history["time"][last_index]
        print("Pick up results at t = ", last_time)
        un = checkpoint.load_function(mesh, "Velocity", idx=last_index)
        etan = checkpoint.load_function(mesh, "Elevation", idx=last_index)

    #print out timestepping history
    print("history in", filename+".h5", "=", history)

    #calculate velocity integral, etanorm and unorm at the last index
    mesh.coordinates.assign(X3)
    etaintegral = assemble(etan * etan * dx)
    print("elevation integral", etaintegral)
    uintegral = assemble(dot(un,un)*dx)
    print("velocity integral", uintegral)
    etanorm = errornorm(etan, etaini)/norm(etaini)
    unorm = errornorm(un, uini, norm_type="Hdiv")/norm(uini, norm_type="Hdiv")
    print('etanorm', etanorm, 'unorm', unorm)
    return history, length

history0, length0 = view_checkpoint(file0)
history1, length1 = view_checkpoint(file1)


# calculate normalised norms in file0 with respect ot file1
print("\n")
print("=== calculate normalised norms in " + file0 + ".h5 with respect to " + file1 + ".h5 ===")
etanorm_list = []
unorm_list = []
for i in range(min(length0, length1)):
    assert(history0["time"][i] == history1["time"][i])
    print("Picking up the results at t = ", history0["time"][i])

    #pickup results
    with CheckpointFile(file0+".h5", 'r') as checkpoint0:
        mesh0 = checkpoint0.load_mesh("mesh")
        X30 = checkpoint0.load_function(mesh0, "coordinates")
        un0 = checkpoint0.load_function(mesh0, "Velocity", idx=i)
        etan0 = checkpoint0.load_function(mesh0, "Elevation", idx=i)
    with CheckpointFile(file1+".h5", 'r') as checkpoint1:
        un1 = checkpoint1.load_function(mesh0, "Velocity", idx=i)
        etan1 = checkpoint1.load_function(mesh0, "Elevation", idx=i)

    #set mesh0
    mesh0.coordinates.assign(X30)

    #print out integrals for debug
    print("<integrals in " + file0 +".h5>")
    etaintegral0 = assemble(etan0 * etan0 * dx)
    uintegral0 = assemble(dot(un0,un0)*dx)
    print("elevation integral", etaintegral0)
    print("velocity integral", uintegral0)

    print("<integrals in " + file1 +".h5>")
    etaintegral1 = assemble(etan1 * etan1 * dx)
    print("elevation integral", etaintegral1)
    uintegral1 = assemble(dot(un1,un1)*dx)
    print("velocity integral", uintegral1)

    #calculate norms
    etanorm = errornorm(etan0, etan1)/norm(etan1)
    unorm = errornorm(un0, un1, norm_type="Hdiv")/norm(un1, norm_type="Hdiv")
    print("<normalised norms>")
    print('etanorm', etanorm, 'unorm', unorm, "\n")
    etanorm_list.append(float(etanorm))
    unorm_list.append(float(unorm))

print("etanorm_list = ", etanorm_list)
print("unorm_list = ", unorm_list)
print("\n")

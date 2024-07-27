from firedrake import *
from firedrake.__future__ import interpolate
import argparse


#get command arguments
parser = argparse.ArgumentParser(description='Creating mesh for Williamson 5 testcase.')
parser.add_argument('--ref_level', type=int, default=3, help='Refinement level of icosahedral grid. Default 4.')
parser.add_argument('--filename', type=str, default='mesh')

args = parser.parse_known_args()
args = args[0]
ref_level = args.ref_level
name = args.filename
print(args)

#parameters
R0 = 6371220.
mesh_degree = 3
mesh = IcosahedralSphereMesh(radius=R0, refinement_level=ref_level,
                                 degree=mesh_degree, name="mesh")
x = SpatialCoordinate(mesh)
global_normal = as_vector([x[0], x[1], x[2]])
mesh.init_cell_orientations(global_normal)

# create checkpoint file and save initial condition
with CheckpointFile(name+".h5", 'w') as checkpoint:
    checkpoint.save_mesh(mesh)
    print("Mesh created and saved in "+name+".h5")

from firedrake import *

base_n = 6
bmesh = UnitSquareMesh(base_n, base_n)
mh = MeshHierarchy(bmesh, 4)
mesh = mh[-1]

V = FunctionSpace(mesh, "BDM", 1)
u = TrialFunction(V)
v = TestFunction(V)

a = (inner(u, v) + inner(div(u), div(v)))*dx
x, y = SpatialCoordinate(mesh)

f = as_vector([0,-0.5*pi*pi*(4*cos(pi*x) - 5*cos(pi*x*0.5) + 2)*sin(pi*y)])

L = inner(f, v)*dx

parameters = {
    "mat_type": "matfree",
    "ksp_type": "fgmres",
    "ksp_monitor_true_residual": None,
    "ksp_converged_reason": None,
    "ksp_atol": 1e-8,
    "ksp_rtol": 1e-8,
    "ksp_max_it": 400,
    "pc_type": "mg",
    "pc_mg_cycle_type": "v",
    "pc_mg_type": "multiplicative",
    "mg_levels_ksp_type": "gmres",
    "mg_levels_ksp_max_it": 3,
    "mg_levels_pc_type": "python",
    "mg_levels_pc_python_type": "firedrake.AssembledPC",
    "mg_levels_assembled_pc_type": "python",
    "mg_levels_assembled_pc_python_type": "firedrake.ASMStarPC",
    "mg_levels_assembled_pc_star_construct_dim": 0,
    "mg_coarse_pc_type": "python",
    "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "mg_coarse_assembled_pc_type": "lu",
    "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
}

u0 = Function(V)
solve(a == L, u0, solver_parameters = parameters)


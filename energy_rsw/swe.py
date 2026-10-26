import firedrake as fd
from firedrake.petsc import PETSc
import irksome
from math import fabs

PETSc.Sys.popErrorHandler()

# physical parameters
R0 = 6371220.
g = fd.Constant(9.8)
Omega = fd.Constant(7.292e-5)
H = fd.Constant(0)

def MGIcosahedralSphereMesh(radius,
                            refinement_level,
                            base_level,
                            degree,
                            distribution_parameters):
    dps = distribution_parameters
    basemesh = fd.IcosahedralSphereMesh(radius=radius,
                                        refinement_level=base_level,
                                        degree=degree,
                                        distribution_parameters=dps)
    del basemesh._radius
    mh = fd.MeshHierarchy(basemesh, refinement_level-base_level)
    meshes = []
    for m in mh:
        X = fd.VectorFunctionSpace(m, "Lagrange", degree)
        new_coords = fd.Function(X).interpolate(m.coordinates)
        x, y, z = new_coords
        r = (x**2 + y**2 + z**2)**0.5
        new_coords = fd.Function(X).interpolate(radius*new_coords/r)
        new_mesh = fd.Mesh(new_coords, name="errormesh")
        x = fd.SpatialCoordinate(new_mesh)
        new_mesh.init_cell_orientations(x)
        meshes.append(new_mesh)

    mh = fd.HierarchyBase(meshes, mh.coarse_to_fine_cells,
                          mh.fine_to_coarse_cells,
                          mh.refinements_per_level, mh.nested)
    return mh[-1]


def w5_ics(u0, D0, b):
    """
    The testcase 5 (flow over a mountain) of Williamson et al 1992
    """

    R00 = fd.Constant(R0)
    mesh = u0.function_space().mesh()
    x, y, z = fd.SpatialCoordinate(mesh)
    u_0 = 20.0  # maximum amplitude of the zonal wind [m/s]
    H.assign(5960.)
    u_max = fd.Constant(u_0)
    u_expr = fd.as_vector([-u_max*y/R00, u_max*x/R00, 0.0])
    eta_expr = - ((R00*Omega*u_max + u_max*u_max/2.0)*(z*z/(R00*R00)))/g
    # Topography.
    rl = fd.pi/9.0
    lambda_x = fd.atan2(y/R00, x/R00)
    lambda_c = -fd.pi/2.0
    phi_x = fd.asin(z/R00)
    phi_c = fd.pi/6.0
    minarg = fd.min_value(pow(rl, 2),
                          pow(phi_x-phi_c, 2) + pow(lambda_x-lambda_c, 2))
    bexpr = 2000.0*(1 - fd.sqrt(minarg)/rl)
    b.interpolate(bexpr)
    u0.interpolate(u_expr)
    D0.interpolate(eta_expr + H - bexpr)


def w6_ics(u0, D0, b, E):
    """
    The testcase 6 (Rossby-Haurwitz wave) of Williamson et al 1992
    """
    mesh = u0.function_space().mesh()
    x, y, z = fd.SpatialCoordinate(mesh)
    lon = fd.atan2(y, x)
    ll = (x**2 + y**2)**0.5
    lat = fd.atan2(z, ll)
    
    def perp(u):
        outward_normals = fd.CellNormal(mesh)
        return fd.cross(outward_normals, u)

    R00 = fd.Constant(R0)
    # code copied from Gusto
    R = fd.Constant(4)
    K = fd.Constant(7.847e-6)  # Frequency parameter, in sec^-1
    w = K
    H0 = fd.Constant(8000.)
    psi = fd.Function(E)
    psiexpr = -R00**2 * w * fd.sin(lat) + \
        R00**2 * K * fd.cos(lat)**R * fd.sin(lat) * fd.cos(R*lon)
    psi.interpolate(psiexpr)
    u_expr = perp(fd.grad(psi))
    u0.project(u_expr)
    # Initialising the depth field
    A = (w / 2) * (2 * Omega + w) * fd.cos(lat)**2 + \
        0.25 * K**2 * fd.cos(lat)**(2 * R) * \
        ((R + 1) * fd.cos(lat)**2 + (2 * R**2 - R - 2) -
         2 * R**2 * fd.cos(lat)**(-2))
    B_frac = (2 * (Omega + w) * K) / ((R + 1) * (R + 2))
    B = B_frac * fd.cos(lat)**R * ((R**2 + 2 * R + 2) -
                                   (R + 1)**2 * fd.cos(lat)**2)
    C = (1 / 4) * K**2 * fd.cos(lat)**(2 * R) * \
        ((R + 1)*fd.cos(lat)**2 - (R + 2))
    Dexpr = H0 + R00**2 * (A + B*fd.cos(lon*R) + C * fd.cos(2 * R * lon))/g
    D0.interpolate(Dexpr)


def gswe(options_dictionary={}):
    # options
    opts = PETSc.Options()
    for k, v in options_dictionary.items():
        opts[k] = v
        
    # mesh
    dps = {"partition": True, "overlap_type":
        (fd.DistributedMeshOverlapType.VERTEX, 2)}
    baserefs = opts.getInt('mesh_baserefs')
    nrefs = opts.getInt('mesh_nrefs')
    meshdeg = opts.getInt('mesh_degree')
    mesh = MGIcosahedralSphereMesh(R0, nrefs, baserefs, meshdeg, dps)

    # spatial quadrature rules
    spqdegree = opts.getInt("quadrature_degree")
    dx = fd.dx(degree=spqdegree)
    dS = fd.dS(degree=spqdegree)

    # three component or two component formulation?
    n_cpts = opts.getInt('stepper_components', 2)
    # spaces
    degree = opts.getInt('functionspaces_degree', 2)
    V = fd.FunctionSpace(mesh, "BDM", degree)
    Q = fd.FunctionSpace(mesh, "DG", degree-1)
    if n_cpts == 2:
        W = fd.VectorFunctionSpace(mesh, "BDM", degree, dim=2)
    elif n_cpts == 3:
        W = V * V * Q
    else:
        raise NotImplementedError

    E = fd.FunctionSpace(mesh, "CG", degree+1)

    # fields
    u0 = fd.Function(V, name="Velocity")
    D0 = fd.Function(Q, name="Layer Depth")
    eta0 = fd.Function(Q, name="Elevation")
    b = fd.Function(Q, name="Topography")
    U0 = fd.Function(W, name="All fields")

    #initial conditions
    testcase = opts.getString('testcase')
    if testcase == "w5":
        w5_ics(u0, D0, b)
    elif testcase == "w6":
        w6_ics(u0, D0, b, E)
    else:
        raise NotImplementedError

    One = fd.Function(Q).assign(1.0)
    H.assign(fd.assemble(D0*dx)/fd.assemble(One*dx))
    assert fabs(fd.assemble((H-D0)*dx)
                )/fd.assemble(One*dx) < 1.0e-7

    if n_cpts == 2:
        #initialising U0
        #component 1 of U0 is u0. component 2 of U0 is G st D = H - div(G)
        WG = V * Q
        uG, phi = fd.TrialFunctions(WG)
        v, q = fd.TestFunctions(WG)
        eqn = (fd.inner(uG, v) - fd.div(v)*phi
               + q*(fd.div(uG) + (D0 - H)))*dx
        v_basis = fd.VectorSpaceBasis(constant=True, comm=fd.COMM_WORLD)
        nullspace = fd.MixedVectorSpaceBasis(WG, [WG.sub(0), v_basis])
        UG = fd.Function(WG)
        hybridparams = {"mat_type": "matfree",
                        "ksp_type": "gmres",
                        "ksp_atol": 0,
                        "ksp_rtol": 1.0e-10,
                        "pc_type": "python", 'pc_python_type':
                        'firedrake.HybridizationPC',
                        'hybridization': {
                            'ksp_type': 'preonly',
                            "ksp_error_if_not_converged": None,
                            'pc_type': 'lu',
                            'pc_factor_mat_solver_type': 'mumps'}
                        }
        fd.solve(fd.lhs(eqn) == fd.rhs(eqn), UG,
                 solver_parameters=hybridparams,
                 nullspace=nullspace)
        uG, p = UG.subfunctions
        U0.interpolate(fd.as_tensor([u0, uG]))
        G = U0[1, :]
        Dtest = fd.Function(Q)
        Dtest.project(H - fd.div(G),
                      form_compiler_parameters
                      = {"quadrature_degree": spqdegree})
        res = fd.norm(Dtest - D0)/fd.norm(One)
        assert res < 1.0e-7, res
    else:
        wu, wG, wD = U0.subfunctions
        wD.interpolate(D0)
        wu.interpolate(u0)
        
    #Shallow Water equations
    def perp(u):
        outward_normals = fd.CellNormal(mesh)
        return fd.cross(outward_normals, u)

    dU = fd.TestFunction(W)
    if n_cpts == 2:
        du = dU[0, :]
        dG = dU[1, :]
        u = U0[0, :]
        G = U0[1, :]
        D = H - fd.div(G)

    elif n_cpts == 3:
        du, dG, dD = fd.TestFunctions(W)
        u, G, D = fd.split(U0)
    
    x = fd.SpatialCoordinate(mesh)
    cx, cy, cz = x
    f = 2*Omega*cz/fd.Constant(R0)
    n = fd.FacetNormal(mesh)
    
    # D = H - div(G)
    # G_t + u*(div(G)-H) = 0
    # therefore
    # D_t = -div(G_t) = div(u*(div(G)-H)) = -div(u*D)
    # ie D_t + div(u*D) = 0
    Dt = irksome.Dt
    F = Dt(G)
    ubar = F/D

    from firedrake import inner, dot, grad, div, sign
    def both(u):
        return 2*fd.avg(u)

    #Time quadrature rules
    time_order = opts.getInt('stepper_time_order', 1)
    s = time_order
    TimeDerivatives = irksome.TimeQuadratureLabel((s-1)*2)
    HamiltonianBits = irksome.TimeQuadratureLabel((s-1)+s*2)

    Upwind = 0.5 * (sign(dot(u, n)) + 1)
    eqn = TimeDerivatives(inner(du, Dt(u))*dx)
    eqn -= inner(perp(grad(inner(du, perp(ubar)))), u)*dx
    eqn += inner(both(perp(n)*inner(du, perp(ubar))), both(Upwind*u))*dS
    eqn += inner(du, f*perp(ubar))*dx
    eqn -= HamiltonianBits(div(du)*(
        inner(u, u)/2
        + g*(D+b)
    )*dx)

    eqn += TimeDerivatives(fd.inner(dG, Dt(G))*dx)
    eqn += HamiltonianBits(fd.inner(dG, -u*D)*dx)

    if n_cpts == 3:
        eqn += TimeDerivatives(dD*(Dt(D) + div(Dt(G)))*dx)

    # vorticity computation
    vort = fd.TrialFunction(E)
    dvort = fd.TestFunction(E)
    vorticity = fd.Function(E, name="Relative Vorticity")
    vort_lhs = vort*dvort*fd.dx
    vort_rhs = -fd.inner(perp(fd.grad(dvort)), u0)*fd.dx
    vort_prob = fd.LinearVariationalProblem(vort_lhs,
                                            vort_rhs,
                                            vorticity)
    vortparams = {'ksp_type': 'preonly',
                  'pc_type': 'lu',
                  "pc_factor_mat_solver_type": "mumps"}
    LVS = fd.LinearVariationalSolver
    vort_solver = LVS(vort_prob,
                      solver_parameters=vortparams)
    
    # timestepper
    scheme_type = opts.getString('stepper_scheme_type', 'cpg')
    if scheme_type == 'cpg':
        quadrature_degree = opts.getInt('stepper_time_quadrature_degree')

        quadrature_scheme = None
        method = irksome.GalerkinCollocationScheme(
            order=time_order,
            stage_type="deriv",
            quadrature_degree=quadrature_degree,
            quadrature_scheme=quadrature_scheme)
    elif scheme_type == 'irk':
        method = irksome.GaussLegendre(time_order)
        eqn = irksome.labeling.as_form(eqn)

    MC = irksome.MeshConstant(mesh)
    dT = MC.Constant(1.)
    t = MC.Constant(0.)

    scheme_J = None
    J = None
    scheme_Jp = None
    Jp = None

    # preconditioning schemes
    if opts.hasName("stepper_schemeJp"):
        schemeJp = opts.getString("stepper_schemeJp")
        if schemeJp == "rana":
            scheme_Jp = irksome.RanaLDScheme(method)
            Jp = irksome.labeling.as_form(eqn)
        if schemeJp == "reducedquadrature":
            scheme_Jp = irksome.as_butcher_tableau(method)
            Jp = irksome.labeling.as_form(eqn)

    # quasiNewton schemes
    if opts.hasName("stepper_schemeJ"):
        schemeJ = opts.getString("stepper_schemeJ")
        if schemeJ == "reducedquadrature":
            scheme_J = irksome.as_butcher_tableau(method)
            J = irksome.labeling.as_form(eqn)

    try:
        mattype = opts.getString("stepper_mat_type")
    except:
        mattype = False
    if mattype:
        sp = {"mat_type": mattype}
    else:
        sp = {}
        
    stepper = irksome.TimeStepper(eqn, method, t, dT, U0,
                                  solver_parameters=sp,
                                  options_prefix="stepper",
                                  scheme_J=scheme_J, J=J,
                                  scheme_Jp=scheme_Jp, Jp=Jp)

    # time parameters
    dt = opts.getScalar('dt')
    tmax = opts.getScalar('tmax')
    nsteps = int(tmax/dt)
    assert fabs(nsteps*dt - tmax) < 1.0e-6*dt, 'tmax is not a multiple of dt'
    dT.assign(dt)

    # output setup
    filename = opts.hasName("filename")
    if filename:
        filename = opts.getString("filename")
        vtkfreq = opts.getInt("vtkfreq", -999)
        chkptfreq = opts.getInt("chkptfreq", -999)
    else:
        vtkfreq = -999
        chkptfreq = -999
    vtk_count = 0
    chkpt_count = 0
    nchk = 0
    if filename and vtkfreq >= 0:
        if n_cpts == 2:
            eta0.interpolate(b-fd.div(U0[1, :]))
        else:
            eta0.interpolate(D0-H+b)
        vort_solver.solve()
        vtkfile = fd.VTKFile(filename+".pvd")
        vtkfile.write(u0, D0, eta0, vorticity)
    if filename and chkpt_count >= 0 and chkptfreq >= 0:
        with fd.CheckpointFile(filename+".h5", 'w') as cfile:
            fields = [u0, D0, eta0]
            for field in fields:
                cfile.save_function(field, idx=0)

    # the time loop
    energy = []
    itcount = 0

    if n_cpts == 2:
        u0.interpolate(U0[0, :])
        eta0.interpolate(b-fd.div(U0[1, :]))
        D0.interpolate(H-fd.div(U0[1, :]))
    else:
        u1, G1, D1 = U0.subfunctions
        u0.interpolate(u1)
        D0.interpolate(D1)
        eta0.interpolate(D1 - H + b)
        energy.append(fd.assemble(0.5*inner(u0, u0)*D*dx
                                  + g*D0*(0.5*D0 + b)*dx))

    for step in fd.ProgressBar('Timestep').iter(range(nsteps)):
        stepper.stages.zero()
        if opts.hasName("stepper_set_atol"):
            # set absolute tolerance automatically to avoid oversolving
            # in last iteration
            F = stepper.solver._ctx._assemble_residual()
            with fd.assemble(F).dat.vec_ro as vec:
                res0 = vec.norm()
            snes_rtol = stepper.solver.snes.rtol
            stepper.solver.snes.ksp.atol = 0.1*snes_rtol*res0

        if step == 0:
            event = "FirstStep"
        else:
            event = "LaterSteps"
        with PETSc.Log.Event(event):
            stepper.advance()
        itcount += stepper.solver.snes.getLinearSolveIterations()
        t.assign(float(t) + float(dt))

        # diagnostics
        if n_cpts == 2:
            u0.interpolate(U0[0, :])
            eta0.interpolate(b-fd.div(U0[1, :]))
            D0.interpolate(H-fd.div(U0[1, :]))
        else:
            u1, G1, D1 = U0.subfunctions
            u0.interpolate(u1)
            D0.interpolate(D1)
            eta0.interpolate(D1 - H)
        energy.append(fd.assemble(0.5*inner(u0, u0)*D*dx
                                  + g*D0*(0.5*D0 + b)*dx))

        # VTK
        vtk_count += 1
        if vtk_count == vtkfreq:
            vort_solver.solve()
            vtkfile.write(u0, D0, eta0, vorticity)
            vtk_count = 0

        # checkpointing
        chkpt_count += 1
        if chkpt_count == chkptfreq:
            nchk += 1
            with fd.CheckpointFile(filename+".h5", 'a') as cfile:
                fields = [u0, D0, eta0]
                for field in fields:
                    cfile.save_function(field, idx=nchk)
            chkpt_count = 0

    PETSc.Sys.Print(f"Time = {(step+1)*dt}")
    return energy, itcount, step


if __name__ == "__main__":
    energy, itcount, steps = gswe()
    energy_err = (energy[-1]-energy[0])/energy[0]
    PETSc.Sys.Print(f"Energy error = {energy_err}.")
    PETSc.Sys.Print(f"Iterations = {itcount}, its per step = {itcount/(steps+1)}")

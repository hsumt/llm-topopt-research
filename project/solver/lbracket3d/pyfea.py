"""FE helpers ported from mfrto/p_norm_stress_constraint/lbracket/simplified_3D_holes.

Original example contributors: A. Guibert, M. Pozzi, M. Bookwala.
Port modification (2026-10-02): failed linear solves raise instead of silently
continuing optimization; unused eigenvalue helpers remain for source fidelity.
"""

from dolfinx import fem, mesh, cpp
from petsc4py import PETSc
from slepc4py import SLEPc
import ufl
import numpy as np
from mpi4py import MPI
from dolfinx.fem.petsc import assemble_vector

# This modules provides some useful functions:
# 1. Subdomain class for boundary conditions
# 2. PETSc linear solver
# 3. SLEPc eigenvalue solver
# 4. Modal Assurance Crierion (MAC)
# 5. Sensitivity analysis

##################################################
# Subdomains
##################################################
class myMat():
    def __init__(self, ui, M_fem, bcs):
        self.ui = ui
        self.M_fem = M_fem
        self.bcs = bcs
    def create(self, A):
        M, N = A.getSize()
        assert M == N

    def mult(self, A, x, y):
        self.action_A(x).copy(y)

    def action_A(self,x):
        ui = self.ui
        M_fem = self.M_fem

        #   Update coefficient ui of the linear form M
        x.copy(ui.vector)
        ui.x.scatter_forward()

        # Compute action of A on x using the linear form M
        y = assemble_vector(M_fem)

        # # Set BC dofs to zero (effectively zeroes rows of A)
        with y.localForm() as y_local:
            fem.set_bc(y_local, self.bcs, scale=0.0)
        y.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)
        return y

class Subdomain:
    def __init__(self, coord, tol):
        self.coord = coord
        self.tol = tol
    def inside(self, x):
        is_inside = np.isclose(x[0], self.coord[0], atol=self.tol[0])
        for i in range(1, len(self.coord)):
            is_inside *= np.isclose(x[i], self.coord[i], atol=self.tol[i])
        return is_inside

############################################################################
# PETSc linear solver
############################################################################

def linear_solver(comm, bilinear, linear, bcs, solution, solver_type, pc_type, nonzeroig=False):
    
    # Assemble bilinear form
    A = fem.petsc.assemble_matrix(bilinear, bcs=bcs)
    A.assemble()

    # Assemble linear form
    b = fem.petsc.create_vector(linear)
    fem.petsc.assemble_vector(b, linear)
    fem.petsc.apply_lifting(b, [bilinear], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, bcs)

    # Create linear solver with preconditioner
    solver = PETSc.KSP().create(comm)
    solver.setTolerances(max_it=10000)
    rank_solver = comm.Get_rank()
    if solver_type == "CG":
        solver.setType(PETSc.KSP.Type.CG) 
    elif solver_type == "PREONLY":
        solver.setType(PETSc.KSP.Type.PREONLY)
    elif solver_type == "GMRES":
        solver.setType(PETSc.KSP.Type.GMRES)
    else:
        if rank_solver == 0: print("Please select a valid solver", flush=True)

    if pc_type == "LU":
        solver.getPC().setType(PETSc.PC.Type.LU) 
    elif pc_type == "ILU":
        solver.getPC().setType(PETSc.PC.Type.ILU)
    elif pc_type == "SOR":
        solver.getPC().setType(PETSc.PC.Type.SOR)
    elif pc_type == "GAMG":
        solver.getPC().setType(PETSc.PC.Type.GAMG)
    else:
        if rank_solver == 0: print("Please select a valid pc", flush=True)


    # Solve
    solver.setInitialGuessNonzero(nonzeroig)
    solver.setOperators(A)
    solver.solve(b, solution.vector)
    solution.x.scatter_forward()

    # Check convergence
    reason = int(solver.getConvergedReason())
    
    # Destroy
    solver.destroy()
    A.destroy()
    b.destroy()

    if reason <= 0:
        raise RuntimeError(f"Linear solver did not converge (PETSc reason {reason})")


def converged(ksp, iter, r_norm):
    rtol, _, _, max_iter = ksp.getTolerances()
    if iter > max_iter:
        return PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT
    r0_norm = ksp.getConvergenceHistory()[0]
    if r_norm / r0_norm < rtol:
        return PETSc.KSP.ConvergedReason.CONVERGED_RTOL
    return PETSc.KSP.ConvergedReason.ITERATING

############################################################################
# SLEPc eigensolver
############################################################################

def eigenfrequency_solver(comm, bilinear, mass, bcs, eigenvector, nev):

    # In FEniCS, the boundary conditions are applied by modifying the stiffness and mass matrices.
    # In particular, the rows corresponding to the boundary conditions are set to zero, except for the diagonal entry, which is set to a custom value (the default is 1.0)
    # When dealing with eigenvalue problems, the diagonal entries will create fictitious eigenvalues equal to the ratio between the diagonal entries of the stiffness and mass matrices.
    # We need to make sure that these fictitious eigenvalues are not in the range of interest, otherwise the eigenvalue solver will return these instead of the actual eigenvalues.
    # Since we are usually interested in the first few eigenvalues (smallest magnitude), we can set the diagonal entries in such a way that the fictitious eigenvalues are very large.
    # The problem is that we do not know a priori the magnitude of the eigenvalues, so we cannot set the diagonal entries to a fixed value.
    # Instead, we first assemble the stiffness matrix with the default diagonal entries of 1.0.
    # Then, we extract the diagonal entries and find the maximum absolute value. Finally, we reassamble the stiffness matrix with the diagonal entries set to the maximum absolute value.
    # We do the same for the mass matrix, but we use the minimum absolute value instead.
    # In this way, we make sure that the fictitious eigenvalues are very large and they will not be returned by the eigenvalue solver.
    # TODO: this is a workaround, but it is not very elegant. Is there a better way to do this?

    # Stiffness matrix
    K_temp = fem.petsc.assemble_matrix(bilinear, bcs=bcs)#, diagonal=1e9)
    K_temp.assemble()
    K_diag = K_temp.getDiagonal()
    K_max = abs(K_diag).max()
    K = fem.petsc.assemble_matrix(bilinear, bcs=bcs, diagonal=K_max[1])
    K.assemble()

    # Mass matrix
    M_temp = fem.petsc.assemble_matrix(mass, bcs=bcs)#, diagonal=1e-3)
    M_temp.assemble()
    M_diag = M_temp.getDiagonal()
    M_min = abs(M_diag).min()
    M = fem.petsc.assemble_matrix(mass, bcs=bcs, diagonal=M_min[1])
    M.assemble()

    # Eigenvalue solver
    eigensolver = SLEPc.EPS().create(comm) # eigenvalue problem solver
    eigensolver.setOperators(K, M) # stiffness and mass matrices
    eigensolver.setDimensions(nev) # number of eigenvalues to compute
    eigensolver.setTolerances(tol=1e-8, max_it=1000) # tolerance and maximum number of iterations
    eigensolver.setType(SLEPc.EPS.Type.KRYLOVSCHUR) # solver type
    eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHEP) # problem type
    eigensolver.getST().setType(SLEPc.ST.Type.SINVERT) # spectral transformation type (shift-and-invert)
    eigensolver.getST().setShift(0.0) # spectral shift value
    eigensolver.setFromOptions() # set options

    # Solve
    eigensolver.solve()

    # Check convergence
    ncv = eigensolver.getConverged()
    if ncv == 0:
        raise Exception("No eigenvalues found")

    # Loop over eigenpairs
    eigenvalues = []
    eigenvectors = []
    eigenfrequencies = []
    for i in range(ncv):

        # Get eigenpair
        eigenvalue = eigensolver.getEigenpair(i, eigenvector.vector)
        eigenvector.x.scatter_forward()

        # Append
        eigenvalues.append(eigenvalue.real)
        eigenvectors.append(eigenvector.vector.array.copy())
        eigenfrequencies.append(np.sqrt(eigenvalue.real) / (2.0 * np.pi))

        # Barrier
        comm.barrier()

    # Destroy
    eigensolver.destroy()
    K_temp.destroy()
    K_diag.destroy()
    K.destroy()
    M_temp.destroy()
    M_diag.destroy()
    M.destroy()

    # Return
    return eigenvalues, eigenvectors, eigenfrequencies

def buckling_solver(comm, bilinear, mass, bcs, eigenvector, nev):

    # Stiffness matrix
    K_temp = fem.petsc.assemble_matrix(bilinear, bcs=bcs)#, diagonal=1e9)
    K_temp.assemble()
    K_diag = K_temp.getDiagonal()
    K_max = abs(K_diag).max()
    K = fem.petsc.assemble_matrix(bilinear, bcs=bcs, diagonal=K_max[1])
    K.assemble()

    # Mass matrix
    M_temp = fem.petsc.assemble_matrix(mass, bcs=bcs)#, diagonal=1e-3)
    M_temp.assemble()
    M_diag = M_temp.getDiagonal()
    M_min = abs(M_diag).min()
    M = fem.petsc.assemble_matrix(mass, bcs=bcs, diagonal=M_min[1])
    M.assemble()

    # Eigenvalue solver
    eigensolver = SLEPc.EPS().create(comm) # eigenvalue problem solver
    eigensolver.setOperators(K, M) # stiffness and mass matrices
    eigensolver.setDimensions(nev) # number of eigenvalues to compute
    eigensolver.setTolerances(tol=1e-8, max_it=1000) # tolerance and maximum number of iterations
    eigensolver.setType(SLEPc.EPS.Type.KRYLOVSCHUR) # solver type
    eigensolver.setProblemType(SLEPc.EPS.ProblemType.GNHEP) # problem type
    eigensolver.getST().setType(SLEPc.ST.Type.SINVERT) # spectral transformation type (shift-and-invert)
    eigensolver.getST().setShift(0.0) # spectral shift value
    eigensolver.setFromOptions() # set options

    # Solve
    eigensolver.solve()

    # Check convergence
    ncv = eigensolver.getConverged()
    if ncv == 0:
        raise Exception("No eigenvalues found")

    # Loop over eigenpairs
    eigenvalues = []
    eigenvectors = []
    for i in range(ncv):

        # Get eigenpair
        eigenvalue = eigensolver.getEigenpair(i, eigenvector.vector)
        eigenvector.x.scatter_forward()

        # Append
        eigenvalues.append(eigenvalue.real)
        eigenvectors.append(eigenvector.vector.array.copy())

        # Barrier
        comm.barrier()

    # Destroy
    eigensolver.destroy()
    K_temp.destroy()
    K_diag.destroy()
    K.destroy()
    M_temp.destroy()
    M_diag.destroy()
    M.destroy()

    # Return
    return eigenvalues, eigenvectors

############################################################################
# Modal Assurance Criterion (MAC)
############################################################################

def modal_assurance_criterion(comm, eigenvectors, eigenvector_ref):

    # Number of eigenvectors
    ncv = len(eigenvectors)

    # Loop over eigenpairs
    mac = np.zeros(ncv)
    coeff_rr = np.dot(eigenvector_ref.vector.array, eigenvector_ref.vector.array) # squared norm of the reference eigenvector
    coeff_rr = comm.allreduce(coeff_rr, op=MPI.SUM) # sum over all processes
    for i in range(ncv):

        # Compute dot products
        coeff_ii = np.dot(eigenvectors[i], eigenvectors[i]) # squared norm of the current eigenvector
        coeff_ir = np.dot(eigenvectors[i], eigenvector_ref.vector.array) # dot product between the current and reference eigenvectors

        # Assemble MAC
        comm.barrier()
        coeff_ii = comm.allreduce(coeff_ii, op=MPI.SUM)
        coeff_ir = comm.allreduce(coeff_ir, op=MPI.SUM)
        mac[i] = coeff_ir**2 / (coeff_ii * coeff_rr)

    # Find maximum MAC
    if comm.Get_rank() == 0:
        idx_loc = np.argmax(mac)
        print("Idx %d has MAC %.4f" % (idx_loc, mac[idx_loc]))
    else:
        idx_loc = None

    # Broadcast index
    idx = comm.bcast(idx_loc, root=0)

    # Return
    return idx

############################################################################
# Sensitivity analysis
############################################################################

# General sensitivity analysis for static problem
def sensitivity(comm, bcs, u, d, J, R, adj, dJ):

    # Nomenclature
    # u: state
    # rho: control
    # J: objective
    # R: residual

    # Partial derivatives with respect to state
    pRpu = ufl.adjoint(ufl.derivative(R, u))
    pJpu = ufl.derivative(J, u)

    # Solve adjoint equation
    bilinear_adj = fem.form(pRpu)
    linear_adj = fem.form(-pJpu)
    linear_solver(comm, bilinear_adj, linear_adj, bcs, adj, "CG", "GAMG", True)
    
    # Continuous form
    # L = J + ufl.action(R, adj)
    # pLpd = ufl.derivative(L, d)
    # pLpd_form = fem.form(pLpd)
    # fem.petsc.assemble_vector(dJ.vector, pLpd_form)

    # Partial derivatives with respect to control
    pRprho = fem.petsc.assemble_matrix(fem.form(ufl.adjoint(ufl.derivative(R, d))))
    pRprho.assemble()
    pJprho = fem.petsc.assemble_vector(fem.form(ufl.derivative(J, d)))
    pJprho.assemble()

    # Total derivative
    dJ.vector.array = pJprho + pRprho * adj.vector

    # Destroy
    pRprho.destroy()
    pJprho.destroy()



    

    






















# Compliance sensitivity
def sensitivity_compliance(u, d, R, dJ):

    # Continuous form
    # pLpd = ufl.derivative(ufl.action(R, u), d)
    # pLpd_form = fem.form(pLpd)
    # fem.petsc.assemble_vector(dJ.vector, pLpd_form)

    # Partial derivatives with respect to control
    pRprho = fem.petsc.assemble_matrix(fem.form(ufl.adjoint(ufl.derivative(R, d))))
    pRprho.assemble()

    # Total derivative
    dJ.vector.array = -pRprho * u.vector

    # Destroy
    pRprho.destroy()

# Frequency sensitivity
def sensitivity_frequency(omega, modal_mass, u, d, R, dJ):

    # Continuous form
    # pLpd = ufl.derivative(ufl.action(R, u), d)
    # pLpd_form = fem.form(pLpd)
    # fem.petsc.assemble_vector(dJ.vector, pLpd_form)
    # dJ.vector.array[:] *= 1.0 / (2.0 * omega * modal_mass) / (2.0 * np.pi) # 2pi is to go from rad/s to Hz

    # Partial derivatives with respect to control
    pRprho = fem.petsc.assemble_matrix(fem.form(ufl.adjoint(ufl.derivative(R, d))))
    pRprho.assemble()

    # Total derivative
    dJ.vector.array = pRprho * u.vector / (2.0 * omega * modal_mass) / (2.0 * np.pi)

    # Destroy
    pRprho.destroy()

# Buckling sensitivity
def sensitivity_buckling(comm, bcs, modal_mass, d, u1, u2, R1, R2, adj, dJ):

    # Nomenclature
    # u1: state 1 (static equilibrium)
    # u2: state 2 (buckling problem)
    # R1: residual 1 (static equilibrium)
    # R2: residual 2 (buckling problem)
    # d: control

    # Partial derivatives with respect to u1
    pR1pu1 = ufl.adjoint(ufl.derivative(R1, u1))
    pR2pu1 = -1.0 / modal_mass * ufl.action(ufl.adjoint(ufl.derivative(R2, u1)), u2)

    # Solve adjoint equation
    adj_bilinear = fem.form(pR1pu1)
    adj_linear = fem.form(-pR2pu1)
    linear_solver(comm, adj_bilinear, adj_linear, bcs, adj)

    # Continuous form
    # pLpd = ufl.derivative(ufl.action(R1, adj) - (1.0 / modal_mass) * ufl.action(R2, u2), d)
    # pLpd_form = fem.form(pLpd)
    # fem.petsc.assemble_vector(dJ.vector, pLpd_form)

    # Partial derivatives with respect to d
    pR1pd = fem.petsc.assemble_matrix(fem.form(ufl.adjoint(ufl.derivative(R1, d))))
    pR1pd.assemble()
    pR2pd = fem.petsc.assemble_matrix(fem.form(ufl.adjoint(ufl.derivative(R2, d))))
    pR2pd.assemble()

    # Assemble sensitivity
    dJ.vector.array = pR1pd * adj.vector - 1.0 / modal_mass * pR2pd * u2.vector

    # Destroy
    pR1pd.destroy()
    pR2pd.destroy()

# Sensitivity mechanism
def sensitivity_mechanism(comm, bcs, u, d, J1, J2, R, adj, dJ):

    # Assemble scalars
    obj1_local = fem.assemble_scalar(fem.form(J1))
    obj1 = comm.allreduce(obj1_local, op=MPI.SUM)
    obj2_local = fem.assemble_scalar(fem.form(J2))
    obj2 = comm.allreduce(obj2_local, op=MPI.SUM)

    # Partial derivatives with respect to state
    pRpu = ufl.adjoint(ufl.derivative(R, u))
    pJ1pu = ufl.derivative(J1, u)

    # Solve adjoint equation
    bilinear_adj = fem.form(pRpu)
    linear_adj = fem.form(-pJ1pu)
    linear_solver(comm, bilinear_adj, linear_adj, bcs, adj)

    # Continuous form
    # pL1pd = ufl.derivative(ufl.action(R, adj), d)
    # pL1pd_form = fem.form(pL1pd)
    # pL2pd = ufl.derivative(ufl.action(R, u), d)
    # pL2pd_form = fem.form(pL2pd)
    # dJ1 = fem.Function(dJ.function_space)
    # dJ2 = fem.Function(dJ.function_space)
    # fem.petsc.assemble_vector(dJ1.vector, pL1pd_form)
    # fem.petsc.assemble_vector(dJ2.vector, pL2pd_form)
    # dJ.vector.array = (1.0 / obj2) * dJ1.vector + (obj1 / obj2**2) * dJ2.vector

    # Partial derivatives with respect to control
    pRprho = fem.petsc.assemble_matrix(fem.form(ufl.adjoint(ufl.derivative(R, d))))
    pRprho.assemble()

    # Total derivative
    dJ.vector.array = pRprho * ((1.0 / obj2) * adj.vector + (obj1 / obj2**2) * u.vector)

    # Destroy
    pRprho.destroy()


def linear_solver_mf(comm, a, L, bcs, solution, V, solver="GMRES", pc="GAMG"):


    b = fem.petsc.assemble_vector(fem.form(L))

    # V = a.function_spaces[0]
    ui = fem.Function(V)
    MK = ufl.action(a, ui)
    K_fem = fem.form(MK)

    fem.set_bc(ui.x.array, bcs, scale=-1)
    fem.petsc.assemble_vector(b, K_fem)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, bcs, scale=0.0)
    b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


    # K = PETSc.Mat()
    # K.create(comm=comm)
    K = PETSc.Mat().createPython([(b.local_size, PETSc.DETERMINE),
                              (b.local_size, PETSc.DETERMINE)], comm=comm)
    # K.setType(PETSc.Mat.Type.PYTHON)
    K.setPythonContext(myMat(ui, K_fem, bcs))
    K.setUp()

    # Create linear solver
    solver = PETSc.KSP().create(comm)
    solver.setOperators(K)
    solver.setType(PETSc.KSP.Type.CG)
    # set mg levels
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.NONE)
    # pc.setMGLevels(5)
    # solver.getPC().setType(pc)
    # solver.setTolerances(rtol=rtol, max_it=max_iter)
    solver.setConvergenceHistory()
    solver.setConvergenceTest(converged)

    solver.solve(b, solution.vector)

    # Set BC values in the solution vectors
    solution.x.scatter_forward()
    with solution.vector.localForm() as y_local:
        fem.set_bc(y_local, bcs, scale=1.0)
    # Check convergence
    # if not solver.converged:
    #     print("WARNING: rank {}: linear solver did not converge (code {})".format(comm.rank, solver.reason), flush=True)

    # Destroy
    solver.destroy()
    K.destroy()
    b.destroy()

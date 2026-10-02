#
# Copyright 2022 H Alicia Kim
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""! @file solver.py
     @brief Define the pde solvers using FEniCS
"""

from fenics import *
from fenics_adjoint import *
import numpy as np
from ufl import nabla_div

class RectangularCuboid(SubDomain):
    """! @brief A derived SubDomain class for defining a subdomain that is a 
    rectangular cuboid (a box). 

    Main use is for defining the domain where body loads are applied.
    """
    def __init__(self, x, y, z, tolx, toly, tolz):
        """! @brief Constructor """
        """! 
        @param x 
            double. x coordinate of the center of the rectangular cuboid
        @param y 
            double. y coordinate of the center of the rectangular cuboid
        @param z 
            double. z coordinate of the center of the rectangular cuboid
        @param tolx 
            double. half-width of the rectangular cuboid in the x direction
        @param toly 
            double. half-width of the rectangular cuboid in the y direction
        @param tolz 
            double. half-width of the rectangular cuboid in the z direction
        """
        # Call parent constructor
        SubDomain.__init__(self)

        # Assign attributes
        self.xc   = x
        self.yc   = y
        self.zc   = z
        self.tolx = tolx
        self.toly = toly
        self.tolz = tolz

    def inside(self, x, on_boundary):
        """! @brief Check if a point is within the rectangular cuboid."""
        """!
        @param x 
            ndarray. x, y, z coordinates of the point
        """
        in_range_x = near(x[0], self.xc, self.tolx)
        in_range_y = near(x[1], self.yc, self.toly)
        in_range_z = near(x[2], self.zc, self.tolz)
        in_region  = in_range_x and in_range_y and in_range_z

        return in_region

class RectangularSurface(RectangularCuboid):
    """! @brief A derived SubDomain class for defining a subdomain that is a 
    rectangular surface. 

    Defines a subdomain that is a rectangular region that lies on a boundary 
    surface, ie, the intersection of a box and the domain boundary. Main use is 
    for defining the domain where boundary conditions are applied.
    """
    def __init__(self, x, y, z, tolx, toly, tolz):
        """! @brief Constructor """
        """!
        @param x
            double. x coordinate of the center of the rectangular region
        @param y 
            double. y coordinate of the center of the rectangular region
        @param z 
            double. z coordinate of the center of the rectangular region
        @param tolx
            double. half-width of the rectangular region in the x direction
        @param toly
            double. half-width of the rectangular region in the y direction
        @param tolz
            double. half-width of the rectangular region in the z direction
        """
        # Call parent constructor, all the attributes will be assigned there
        RectangularCuboid.__init__(self, x, y, z, tolx, toly, tolz)

    def inside(self, x, on_boundary):
        """! @brief Check if a point is within a rectangular region defined on 
        a boundary surface. """
        """!
        @param x 
            ndarray. x, y, z coordinates of the point
        """
        # Check if the point lies within the rectangular cuboid
        in_region  = RectangularCuboid.inside(self, x, on_boundary)

        # The point must lie both on the boundary and the in the rectangular
        # cuboid
        return on_boundary and in_region


class LinElasticPde:
    """! @brief Class for the linear elasticity pde using FEniCS. 
    
    The linear elasticity pde is defined in the variational formulation as 
    a(u,v) = L(v) where a(u,v) = integral (sigma(u) : grad(v)) and 
    L(v) = integral dot(f, v). See Section 3.3 The equations of linear 
    elasticity of 
    https://link.springer.com/content/pdf/10.1007%2F978-3-319-52462-7.pdf for
    further details. Note that traction forces are not currently supported.
    """
    
    def __init__(self, nelx, nely, nelz, lx, ly, lz, E, v, rho_min, penal=0):
        """! @brief Constructor."""
        """! Note that the ratios between the number of elements should be the 
        same as the ratios between the lengths. E.g., 2x2x1 [m]  with a mesh 
        size of 80x80x40."""
        """!
        @param nelx 
            int. Number of elements in the x direction
        @param nely 
            int. Number of elements in the y direction
        @param nelz 
            int. Number of elements in the z direction
        @param lx 
            double. Length of the domain in the x direction
        @param ly 
            double. Length of the domain in the x direction
        @param lz 
            double. Length of the domain in the x direction
        @param E 
            double. Young's (elasticity) modulus
        @param v 
            double. Poisson ratio
        @param rho_min 
            double. Minimum density value for void elements
        @param penal
            double. Penalization parameter for SIMP. If 0 no penalization is
            applied (default)
        """
        # Assign attributes
        self.nelx    = nelx
        self.nely    = nely
        self.nelz    = nelz
        self.nelem   = nelx*nely*nelz # number of elements in mesh
        self.ndofs   = 3*(nelx+1)*(nely+1)*(nelz+1) # number of dofs in mesh
        self.mu      = Constant(E / (2*(1+v)))         # lame parameter
        self.lmbda   = Constant(E*v / ((1+v)*(1-2*v))) # lame parameter
        self.rho_min = rho_min
        self.bcs     = [] # empty list for boundary conditions
        self.f       = None # magnitude of the body load
        self.L       = None # linear form, L(v)
        self.a       = None # bilinear form, a(u, v)
        self.penal   = Constant(penal)

        # Create mesh
        temp_mesh = BoxMesh.create(
            [Point(0.0, 0.0, 0.0), Point(lx, ly, lz)], # define opposing corners
            [nelx, nely, nelz], # number of elements in each direction
            CellType.Type.hexahedron)
        self.mesh = Mesh(temp_mesh)

        # Select all the elements in the mesh; this will be used later for
        # defining the domain for applying the body load
        self.body_domain = MeshFunction("size_t", self.mesh,
            self.mesh.topology().dim(), 0)

        # Create a vector-valued finite element function space, i.e the test
        # space or variation space. This will be used to generate vector fields
        # (i.e. displacement).
        # Uses 'CG' (Lagrange) elements of degree 1 (linear)
        self.V = VectorFunctionSpace(self.mesh, "CG", 1)

        # Create a scalar finite element function space. This will be used to
        # generate scalar fields (i.e. density)
        self.D = FunctionSpace(self.mesh, "DG", 0) # Discontinuous elements degree 0

        # Define functions
        self.u       = TrialFunction(self.V) # trial function
        self.v       = TestFunction(self.V)  # test (variation) function
        self.u_sol   = Function(self.V)      # function for defining the solution
        self.density = Function(self.D)      # density field

    def add_clamped_bc(self, x, y, z, tolx, toly, tolz, valx, valy, valz):
        """! @brief Add a clamped (Dirichlet) boundary condition within a 
        rectangular region on a boundary surface."""
        """!
        @param x
            double. x coordinate of the center of the rectangular region
        @param y
            double. y coordinate of the center of the rectangular region
        @param z
            double. z coordinate of the center of the rectangular region
        @param tolx
            double. half-width of the rectangular region in the x direction
        @param toly
            double. half-width of the rectangular region in the y direction
        @param tolz
            double. half-width of the rectangular region in the z direction
        @param valx
            double. Value of the prescribed displacement in the x direction
        @param valy
            double. Value of the prescribed displacement in the y direction
        @param valz
            double. Value of the prescribed displacement in the z direction
        """
        # Define domain of the rectangular region
        subdomain = RectangularSurface(x, y, z, tolx, toly, tolz)

        # Define dirichlet boundary condition
        bc_temp = DirichletBC(self.V, Constant((valx, valy, valz)), subdomain)
        self.bcs.append(bc_temp)

    def add_x_roller_bc(self, x, y, z, tolx, toly, tolz, val):
        """! @brief Add a roller (Dirichlet) boundary condition within a 
        rectangular region on a boundary surface."""
        """! The normal of the roller surface is in the x
        direction, ie, the dofs are fixed in the x direction and free to
        translate in the y, z directions."""
        """!
        @param x
            double. x coordinate of the center of the rectangular region
        @param y
            double. y coordinate of the center of the rectangular region
        @param z
            double. z coordinate of the center of the rectangular region
        @param tolx
            double. half-width of the rectangular region in the x direction
        @param toly
            double. half-width of the rectangular region in the y direction
        @param tolz
            double. half-width of the rectangular region in the z direction
        @param val
            double. Value of the prescribed displacement in the x direction
        """
        # Define domain of the rectangular region
        subdomain = RectangularSurface(x, y, z, tolx, toly, tolz)

        # Define dirichlet boundary condition
        bc_temp = DirichletBC(self.V.sub(0), Constant((val)), subdomain)
        self.bcs.append(bc_temp)

    def add_y_roller_bc(self, x, y, z, tolx, toly, tolz, val):
        """! @brief Add a roller (Dirichlet) boundary condition within a 
        rectangular region on a boundary surface.""" 
        """! The normal of the roller surface is in the y direction, ie, the 
        dofs are fixed in the y direction and free to translate in the x, z 
        directions."""
        """!
        @param x
            double. x coordinate of the center of the rectangular region
        @param y
            double. y coordinate of the center of the rectangular region
        @param z
            double. z coordinate of the center of the rectangular region
        @param tolx
            double. half-width of the rectangular region in the x direction
        @param toly
            double. half-width of the rectangular region in the y direction
        @param tolz
            double. half-width of the rectangular region in the z direction
        @param val
            double. Value of the prescribed displacement in the y direction
        """
        # Define domain of the rectangular region
        subdomain = RectangularSurface(x, y, z, tolx, toly, tolz)

        # Define dirichlet boundary condition
        bc_temp = DirichletBC(self.V.sub(1), Constant((val)), subdomain)
        self.bcs.append(bc_temp)

    def add_z_roller_bc(self, x, y, z, tolx, toly, tolz, val):
        """! @brief Add a roller (Dirichlet) boundary condition within a 
        rectangular region on a boundary surface."""
        """! The normal of the roller surface is in the z direction, ie, the 
        dofs are fixed in the z direction and free to translate in the x, y 
        directions."""
        """!
        @param x
            double. x coordinate of the center of the rectangular region
        @param y
            double. y coordinate of the center of the rectangular region
        @param z
            double. z coordinate of the center of the rectangular region
        @param tolx
            double. half-width of the rectangular region in the x direction
        @param toly
            double. half-width of the rectangular region in the y direction
        @param tolz
            double. half-width of the rectangular region in the z direction
        @param val
            double. Value of the prescribed displacement in the z direction
        """
        # Define domain of the rectangular region
        subdomain = RectangularSurface(x, y, z, tolx, toly, tolz)

        # Define dirichlet boundary condition
        bc_temp = DirichletBC(self.V.sub(2), Constant((val)), subdomain)
        self.bcs.append(bc_temp)

    # TODO: create a function that can apply multiple body loads?
    def apply_body_load(self, x, y, z, tolx, toly, tolz, valx, valy, valz):
        """! @brief Apply a constant body load f within a rectangular cuboid."""
        """! Note that this function can only apply a single body load. If the 
        function is called twice, the body load defined in the the second 
        function call will overwrite the first body load."""
        """!
        @param x
            double. x coordinate of the center of the rectangular cuboid
        @param y
            double. y coordinate of the center of the rectangular cuboid
        @param z
            double. z coordinate of the center of the rectangular cuboid
        @param tolx
            double. half-width of the rectangular cuboid in the x direction
        @param toly
            double. half-width of the rectangular cuboid in the y direction
        @param tolz
            double. half-width of the rectangular cuboid in the z direction
        @param valx
            double. Value of the force in the x direction
        @param valy
            double. Value of the force in the y direction
        @param valz
            double. Value of the force in the z direction
        """
        # Mark all the elements in the domain as 0
        self.body_domain.set_all(0)

        # Mark all the elements within the defined rectangular cuboid as 1
        cuboid = RectangularCuboid(x, y, z, tolx, toly, tolz)
        cuboid.mark(self.body_domain, 1)

        # Assign the magnitude of the body load
        self.f = Constant((valx, valy, valz))

    def define_linear_form(self):
        """! @brief  Defines the linear form L(v) = integral dot (f, v)."""
        """! Note that apply_body_load must be called before calling this 
        function."""
        # Define new measure associated with the defined load_domain
        dx = Measure("dx")(subdomain_data=self.body_domain)

        # Define the linear form. By using dx(1), we are integrating only over
        # the elements that were marked 1 in load_domain
        self.L = dot(self.f, self.v) * dx(1)

    def sigma(self, u):
        """! @brief Stress function."""
        """! Helper function for defining the bilinear form a(u,v)."""
        """!
        @param u
            Dolfin function. Function that describes the displacement vector field
        """
        n = len(u) # size of u
        return self.lmbda*nabla_div(u)*Identity(n) + 2*self.mu*self.epsilon(u)

    def epsilon(self, u):
        """! @brief Strain function."""
        """! Helper function for defining the stress function."""
        """!
        @param u 
            Dolfin function. Function that describes the displacement vector field
        """
        return 0.5*(nabla_grad(u) + nabla_grad(u).T)

    def define_bilinear_form(self, densities):
        """! @brief Defines the bilinear form."""
        """! Defines the bilinear form a(u, v) = integral (sigma(u) : grad(v)),
        where the stress, sigma, is multiplied by the element density."""
        """!
        @param densities
            ndarray. Densities for each element of the FEniCS mesh. FEniCS 
            elements are ordered going first through the x direction, then y 
            direction, and then z direction.
        """
        # Copy densities to FEniCS vectors and apply minimum density value.
        nelem = self.nelx*self.nely*self.nelz
        self.density.vector()[:] = np.zeros(nelem)
        for i in range(nelem):
            self.density.vector()[i] = self.rho_min*(1-densities[i]) + densities[i]

        # Compute a(u, v)
        if float(self.penal) == 0:
            self.a = inner(self.density*self.sigma(self.u), nabla_grad(self.v)) * dx
        else:
            self.a = inner((self.density**self.penal)*self.sigma(self.u), 
                nabla_grad(self.v)) * dx

    def solve(self, rel_tol=1e-5, solver_method='cg', preconditioner='ilu'):
        """! @brief Solve the linear elasticity pde as a linear system A*u = b."""
        """!
        @param rel_tol
            double. Relative tolerance for the preconditioner

        @param solver_method
            str. Name of FEniCS solver. Use list_linear_solver_methods() to see 
            an up-to-date list of the available solvers for your FEniCS
            installation

        @param solver_method
            str. Name of FEniCS preconditioner. Use
            list_linear_solver_preconditioners() to see an up-to-date list of
            the available preconditioners for your FEniCS installation
        """
        # Assemble the system
        self.A, self.b = assemble_system(self.a, self.L, self.bcs)

        # Set parameters and create solver
        solver = KrylovSolver(solver_method, preconditioner)
        prm = solver.parameters
        # TODO: fix syntax for setting parameters. The following gives an error.
        # prm.absolute_tolerance = 1e-11
        # prm.relative_tolerance = rel_tol
        # prm.maximum_iterations = 500

        # Solve the variational problem
        solver.solve(self.A, self.u_sol.vector(), self.b)

    def get_residual(self):
        """! @brief Get the residual R = A*u - b."""
        """! The solve() function should be called before calling this function."""
        """!
        @return
            ndarray. Residual array
        """
        # Convert to numpy arrays
        A_np = self.A.array()
        u_np = self.u_sol.vector().get_local()
        b_np = self.b.get_local()

        res = np.dot(A_np, u_np) - b_np

        return res

    def get_solution(self):
        """! Get the solution of the pde at the nodes, ie, the nodal 
        displacement."""
        """! The solve() function should be called before calling this function. 
        See section 5.4.1 of
        https://link.springer.com/content/pdf/10.1007%2F978-3-319-52462-7.pdf
        for information about the ordering of degrees of freedom."""
        """! 
        @return
            ndarray. Displacement vector
        """
        return self.u_sol.vector().get_local()

    def jacobian_vector_product(self, dv):
        """! @brief Calculate the product between the Jacobian and some vector 
        dv.""" 
        """! The Jacobian is the matrix of of partial derivatives 
        J = [pR/pd pR/pf pR/pu] where pR/pd is the partial derivative of the 
        residual wrt the design variable (densities), pR/pf is the partial wrt 
        the body load, and pR/pu is the partial wrt the state variable 
        (displacement). Note that solve() should be called before calling this 
        function."""
        """!
        @param dv
            ndarray. Vector to multipy the Jacobian by. Should be of size 
            (n+2*m)x1 where n is the number of design variables (i.e. element 
            densities) and m is the number of degrees of freedoms.
        @return
            ndarray. Product of the Jacobian and the given vector.
        """
        # Define the residual functional
        R = action(self.a, self.u_sol) - self.L

        # Calculate the partials of the residual
        pRpd = assemble(derivative(R, self.density))
        proj_f = project(self.f, self.V)
        pRpf = assemble(derivative(R, proj_f))
        pRpu = assemble(derivative(R, self.u_sol))

        # Calculate the products
        n = self.nelem
        m = self.ndofs
        pRpdv = np.dot(pRpd.array(), dv[0:n,:])
        pRpfv = np.dot(pRpf.array(), dv[n:n+m,:])
        pRpuv = np.dot(pRpf.array(), dv[n+m:n+2*m,:])

        return pRpdv + pRpfv + pRpuv

    def jacobian_transpose_vector_product(self, dv):
        """! @brief Calculate the product between the transpose of the Jacobian 
        and some vector dv."""
        """! The Jacobian is the matrix of of partial derivatives J =
        [pR/pd pR/pf pR/pu] where pR/pd is the partial derivative of the
        residual wrt the design variable (densities), pR/pf is the partial wrt
        the body load, and pR/pu is the partial wrt the state variable
        (displacement). The transpose of the Jacobian would therefore be J^T =
        [(pR/pd)^T (pR/pf)^T (pR/pu)^T]. Note that solve() should be called
        before calling this function."""
        """!
        @param dv
            ndarray. Vector to multipy the transpose of the Jacobian by. Should 
            be of size mx1 where m is the number of degrees of freedoms.
        @return
            ndarray. Product of the transpose of the Jacobian and the given 
            vector.
        """
        # Define the residual functional
        R = action(self.a, self.u_sol) - self.L

        # Calculate the partials of the residual
        pRpd = assemble(derivative(R, self.density))
        proj_f = project(self.f, self.V)
        pRpf = assemble(derivative(R, proj_f))
        pRpu = assemble(derivative(R, self.u_sol))

        # Calculate the products
        n = self.nelem
        m = self.ndofs
        pRpdv = np.dot(np.transpose(pRpd.array()), dv)
        pRpfv = np.dot(np.transpose(pRpf.array()), dv)
        pRpuv = np.dot(np.transpose(pRpf.array()), dv)

        return np.concatenate((pRpdv, pRpfv, pRpuv), axis=0)

class PoissonPde:
    """! @brief Class for the Poisson problem pde using FEniCS. 

    The Poisson pde is defined in the variational formulation as a(u,v) = L(v) 
    where a(u,v) = integral dot(grad(u), grad(v)) and L(v) = integral (fv). See 
    Section 2 Fundamentals: Solving the Poisson equation of
    https://link.springer.com/content/pdf/10.1007%2F978-3-319-52462-7.pdf for
    further details.
    """

    def __init__(self, nelx, nely, nelz, lx, ly, lz, kappa, rho_min, penal=0):
        """! @brief  Constructor.""" 
        """! Note that the ratios between the number of elements should
        be the same as the ratios between the lengths. E.g., 2x2x1 [m] with a
        mesh size of 80x80x40."""
        """
        @param nelx
            int. Number of elements in the x direction
        @param nely
            int. Number of elements in the y direction
        @param nelz
            int. Number of elements in the z direction
        @param lx
            double. Length of the domain in the x direction
        @param ly
            double. Length of the domain in the x direction
        @param lz
            double. Length of the domain in the x direction
        @param kappa
            double. Thermal conductivity coefficient
        @param rho_min
            double. Minimum density value for void elements
        @param penal
            double. Penalization parameter for SIMP. If 0 no penalization is
            applied (default)
        """
        # Assign attributes
        self.nelx    = nelx
        self.nely    = nely
        self.nelz    = nelz
        self.nelem   = nelx*nely*nelz # number of elements in mesh
        self.ndofs   = (nelx+1)*(nely+1)*(nelz+1) # number of dofs in mesh
        self.kappa   = Constant(kappa)
        self.rho_min = rho_min
        self.bcs     = [] # empty list for boundary conditions
        self.f       = None # magnitude of the body load
        self.L       = None # linear form, L(v)
        self.a       = None # bilinear form, a(u, v)
        self.penal   = Constant(penal)

        # Create mesh
        temp_mesh = BoxMesh.create(
            [Point(0.0, 0.0, 0.0), Point(lx, ly, lz)], # define opposing corners
            [nelx, nely, nelz], # number of elements in each direction
            CellType.Type.hexahedron)
        self.mesh = Mesh(temp_mesh)

        # Select all the elements in the mesh; this will be used later for
        # defining the domain for applying the body load
        self.body_domain = MeshFunction("size_t", self.mesh,
            self.mesh.topology().dim(), 0)

        # Create a scalar finite element function space, i.e the test space or
        # variation space. This will be used to generate scalar fields
        # (i.e. temperature).
        # Uses 'CG' (Lagrange) elements of degree 1 (linear)
        self.V = FunctionSpace(self.mesh, "CG", 1)

        # Create a scalar finite element function space. This will be used to
        # generate scalar fields (i.e. density)
        self.D = FunctionSpace(self.mesh, "DG", 0) # Discontinuous elements degree 0

        # Define functions
        self.u       = TrialFunction(self.V) # trial function
        self.v       = TestFunction(self.V)  # test (variation) function
        self.u_sol   = Function(self.V)      # function for defining the solution
        self.density = Function(self.D)      # density field

    def add_temperature_bc(self, x, y, z, tolx, toly, tolz, val):
        """! @brief Add a Dirichlet boundary condition, ie, prescribed 
        temperature, within a rectangular region on a boundary surface."""
        """!
        @param x
            double. x coordinate of the center of the rectangular region
        @param y
            double. y coordinate of the center of the rectangular region
        @param z
            double. z coordinate of the center of the rectangular region
        @param tolx
            double. half-width of the rectangular region in the x direction
        @param toly
            double. half-width of the rectangular region in the y direction
        @param tolz
            double. half-width of the rectangular region in the z direction
        @param val
            double. Value of the prescribed temperature
        """
        # Define domain of the rectangular region
        subdomain = RectangularSurface(x, y, z, tolx, toly, tolz)

        # Define dirichlet boundary condition
        bc_temp = DirichletBC(self.V, Constant(val), subdomain)
        self.bcs.append(bc_temp)

    # TODO: create a function that can apply multiple body loads?
    def apply_body_load(self, x, y, z, tolx, toly, tolz, val):
        """! @brief Apply a constant body load f within a rectangular cuboid."""
        """! Note that this function can only apply a single body load. If the 
        function is called twice, the body load defined in the the second 
        function call will overwrite the first body load."""
        """!
        @param x
            double. x coordinate of the center of the rectangular cuboid
        @param y
            double. y coordinate of the center of the rectangular cuboid
        @param z
            double. z coordinate of the center of the rectangular cuboid
        @param tolx
            double. half-width of the rectangular cuboid in the x direction
        @param toly
            double. half-width of the rectangular cuboid in the y direction
        @param tolz
            double. half-width of the rectangular cuboid in the z direction
        @param val
            double. Value of the load
        """
        # Mark all the elements in the domain as 0
        self.body_domain.set_all(0)

        # Mark all the elements within the defined rectangular cuboid as 1
        cuboid = RectangularCuboid(x, y, z, tolx, toly, tolz)
        cuboid.mark(self.body_domain, 1)

        # Assign the magnitude of the body load
        self.f = Constant((val))

    def define_linear_form(self):
        """! @brief Defines the linear form L(v) = integral dot (f, v).""" 
        """! Note that apply_body_load or must be called before calling this 
        function."""
        """!
        @param densities
            ndarray. Densities for each element of the FEniCS mesh. FEniCS elements are
            ordered going first through the x direction, then y direction, and
            then z direction.
        """
        # Define new measure associated with the defined load_domain
        dx = Measure("dx")(subdomain_data=self.body_domain)

        # Define the linear form. By using dx(1), we are integrating only over
        # the elements that were marked 1 in load_domain
        self.L = self.f * self.v * dx(1)

    def define_bilinear_form(self, densities):
        """! @brief Defines the bilinear form."""
        """! Defines the bilinear form a(u, v) = integral (sigma(u) : grad(v)),
        where the stress, sigma, is multiplied by the element density."""
        """!
        @param densities
            ndarray. Densities for each element of the FEniCS mesh. FEniCS 
            elements are ordered going first through the x direction, then y 
            direction, and then z direction.
        """
        # Copy densities to FEniCS vectors and apply minimum density value.
        nelem = self.nelx*self.nely*self.nelz
        self.density.vector()[:] = np.zeros(nelem)
        for i in range(nelem):
            self.density.vector()[i] = self.rho_min*(1-densities[i]) + densities[i]

        # Compute a(u, v)
        if float(self.penal) == 0:
            self.a = inner(self.kappa*self.density*grad(self.u), grad(self.v)) * dx
        else:
            self.a = inner(self.kappa*(self.density**self.penal)*grad(self.u), 
                grad(self.v)) * dx

    def solve(self, rel_tol=1e-5, solver_method='cg', preconditioner='ilu'):
        """! @brief Solve the linear elasticity pde as a linear system A*u = b."""
        """!
        @param rel_tol
            double. Relative tolerance for the preconditioner

        @param solver_method
            str. Name of FEniCS solver. Use list_linear_solver_methods() to see an
            up-to-date list of the available solvers for your FEniCS
            installation

        @param solver_method
            str. Name of FEniCS preconditioner. Use
            list_linear_solver_preconditioners() to see an up-to-date list of
            the available preconditioners for your FEniCS installation
        """
        # Assemble the system
        self.A, self.b = assemble_system(self.a, self.L, self.bcs)

        # Set parameters and create solver
        solver = KrylovSolver(solver_method, preconditioner)
        prm = solver.parameters
        # TODO: fix syntax for setting parameters. The following gives an error.
        # prm.absolute_tolerance = 1e-11
        # prm.relative_tolerance = rel_tol
        # prm.maximum_iterations = 500

        # Solve the variational problem
        solver.solve(self.A, self.u_sol.vector(), self.b)

    def get_residual(self):
        """! @brief  Get the residual R = A*u - b."""
        """! The solve() function should be called before calling this function."""
        """!
        @return
            ndarray. Residual array
        """
        # Convert to numpy arrays
        A_np = self.A.array()
        u_np = self.u_sol.vector().get_local()
        b_np = self.b.get_local()

        res = np.dot(A_np, u_np) - b_np

        return res

    def get_solution(self):
        """! @brief Get the solution of the pde at the nodes, ie, the nodal 
        displacement."""
        """! The solve() function should be called before calling this function. 
        See section 5.4.1 of
        https://link.springer.com/content/pdf/10.1007%2F978-3-319-52462-7.pdf
        for information about the ordering of degrees of freedom."""
        """
        @return
            ndarray. Displacement vector
        """
        return self.u_sol.vector().get_local()

    def jacobian_vector_product(self, dv):
        """! @brief Calculate the product between the Jacobian and some vector 
        dv.""" 
        """! The Jacobian is the matrix of of partial derivatives 
        J = [pR/pd pR/pf pR/pu] where pR/pd is the partial derivative of the 
        residual wrt the design variable (densities), pR/pf is the partial wrt 
        the body load, and pR/pu is the partial wrt the state variable 
        (displacement). Note that solve() should be called before calling this 
        function."""
        """!
        @param dv
            ndarray. Vector to multipy the Jacobian by. Should be of size (n+2*m)x1
            where n is the number of design variables (i.e. element densities)
            and m is the number of degrees of freedoms.
        @return
            ndarray. Product of the Jacobian and the given vector.
        """
        # Define the residual functional
        R = action(self.a, self.u_sol) - self.L

        # Calculate the partials of the residual
        pRpd = assemble(derivative(R, self.density))
        proj_f = project(self.f, self.V)
        pRpf = assemble(derivative(R, proj_f))
        pRpu = assemble(derivative(R, self.u_sol))

        # Calculate the products
        n = self.nelem
        m = self.ndofs
        pRpdv = np.dot(pRpd.array(), dv[0:n,:])
        pRpfv = np.dot(pRpf.array(), dv[n:n+m,:])
        pRpuv = np.dot(pRpf.array(), dv[n+m:n+2*m,:])

        return pRpdv + pRpfv + pRpuv

    def jacobian_transpose_vector_product(self, dv):
        """! @brief Calculate the product between the transpose of the Jacobian 
        and some vector dv.""" 
        """! The Jacobian is the matrix of of partial derivatives 
        J = [pR/pd pR/pf pR/pu] where pR/pd is the partial derivative of the
        residual wrt the design variable (densities), pR/pf is the partial wrt
        the body load, and pR/pu is the partial wrt the state variable
        (displacement). The transpose of the Jacobian would therefore be J^T =
        [(pR/pd)^T (pR/pf)^T (pR/pu)^T]. Note that solve() should be called
        before calling this function."""
        """!
        @param dv
            ndarray. Vector to multipy the transpose of the Jacobian by. Should 
            be of size mx1 where m is the number of degrees of freedoms.
        @return
            ndarray. Product of the transpose of the Jacobian and the given 
            vector.
        """
        # Define the residual functional
        R = action(self.a, self.u_sol) - self.L

        # Calculate the partials of the residual
        pRpd = assemble(derivative(R, self.density))
        proj_f = project(self.f, self.V)
        pRpf = assemble(derivative(R, proj_f))
        pRpu = assemble(derivative(R, self.u_sol))

        # Calculate the products
        n = self.nelem
        m = self.ndofs
        pRpdv = np.dot(np.transpose(pRpd.array()), dv)
        pRpfv = np.dot(np.transpose(pRpf.array()), dv)
        pRpuv = np.dot(np.transpose(pRpf.array()), dv)

        return np.concatenate((pRpdv, pRpfv, pRpuv), axis=0)
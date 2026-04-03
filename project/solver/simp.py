"""
simp.py  —  SIMP Topology Optimisation built on DOLFINx
================================================================
Milestone 1 : DOLFINx cantilever setup  (mesh, V, BCs, weak form)
Milestone 2 : SIMP density loop + sensitivity  (E(ρ), ∂C/∂ρ, OC)
Milestone 3 : Density filter + convergence metrics
Milestone 4 : Density field plotting

Usage
-----
    python simp.py                         # cantilever, default params
    python simp.py --geometry lbracket
    python simp.py --nelx 120 --nely 40 --volfrac 0.4

Dependencies: dolfinx, ufl, mpi4py, petsc4py, numpy, scipy, matplotlib
"""

import argparse
import os
import numpy as np
import matplotlib

from schema import ProblemSpec
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import imageio

import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem, assemble_matrix, assemble_vector
import dolfinx.fem.petsc as fem_petsc

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


# ══════════════════════════════════════════════════════════════════════════════
# MILESTONE 1 — DOLFINx cantilever setup
# ══════════════════════════════════════════════════════════════════════════════

def build_mesh(nelx: int, nely: int, Lx: float = 1.0, Ly: float = 1.0):
    """
    Structured quadrilateral mesh over [0,Lx] x [0,Ly].
    Column-major convention: x is the long axis, y is the short axis.
    """
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([Lx, Ly])],
        [nelx, nely],
        cell_type=mesh.CellType.quadrilateral,
    )
    return domain


def build_function_spaces(domain):
    """
    V  : vector Lagrange P1 — displacement field
    Q  : scalar DG0        — element-wise density (one value per cell)
    """
    V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("DG", 0))
    return V, Q


def build_bcs_cantilever(V, domain, Lx):
    """Fix the left edge (x=0) in both components."""
    def left_boundary(x):
        return np.isclose(x[0], 0.0)

    fixed_dofs = fem.locate_dofs_geometrical(V, left_boundary)
    zero = np.zeros(domain.geometry.dim, dtype=default_scalar_type)
    bc   = fem.dirichletbc(zero, fixed_dofs, V)
    return [bc]


def build_bcs_lbracket(V, domain, Lx, Ly):
    """
    L-bracket: fix the entire top edge (y = Ly).
    Load applied separately at the right mid-height edge.
    """
    def top_boundary(x):
        return np.isclose(x[1], Ly)

    fixed_dofs = fem.locate_dofs_geometrical(V, top_boundary)
    zero = np.zeros(domain.geometry.dim, dtype=default_scalar_type)
    bc   = fem.dirichletbc(zero, fixed_dofs, V)
    return [bc]


def build_traction_measure(domain, fdim, boundary_fn, tag: int = 1):
    facets = mesh.locate_entities_boundary(domain, fdim, boundary_fn)
    
    # Add this line to sort the facets!
    sort_idx = np.argsort(facets)
    
    markers = np.full_like(facets, tag)
    facet_tags = mesh.meshtags(domain, fdim, facets[sort_idx], markers[sort_idx])
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    return ds


def material_constants(E: float = 1.0, nu: float = 0.3):
    """
    Plane-stress Lamé parameters.
    λ = E·ν / (1−ν²),  μ = E / (2(1+ν))
    """
    mu    = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / (1.0 - nu ** 2)
    return mu, lmbda


def epsilon(u):
    return ufl.sym(ufl.grad(u))


def sigma_expr(u, mu, lmbda, dim):
    return lmbda * ufl.tr(epsilon(u)) * ufl.Identity(dim) + 2.0 * mu * epsilon(u)


# ══════════════════════════════════════════════════════════════════════════════
# MILESTONE 2 — SIMP density loop + sensitivity
# ══════════════════════════════════════════════════════════════════════════════

E_MIN = 1e-9   # void stiffness (keeps K non-singular)


def penalised_E(rho_vals: np.ndarray, penal: float) -> np.ndarray:
    """
    SIMP interpolation:  E(ρ) = E_min + (1 − E_min) · ρ^p
    Returns a 1-D array of length n_cells.
    """
    return E_MIN + (1.0 - E_MIN) * rho_vals ** penal


def solve_elasticity(domain, V, bcs, rho_fn: fem.Function,
                     penal: float, mu0: float, lmbda0: float,
                     traction_load,       # (T_vector, ds_measure, tag)
                     ):
    """
    Assemble and solve the penalised linear elasticity problem.

    The element stiffness is scaled by E(ρ) via a DG0 coefficient, which
    multiplies the full bilinear form — equivalent to the standard SIMP
    element-wise interpolation for a P1 discretisation on quads.

    Parameters
    ----------
    traction_load : (fem.Constant, ds_measure, int tag)

    Returns
    -------
    uh : fem.Function   displacement solution
    """
    dim   = domain.geometry.dim
    u, v  = ufl.TrialFunction(V), ufl.TestFunction(V)

    # Penalised stiffness coefficient — spatially-varying DG0 field
    E_coeff = fem.Function(rho_fn.function_space)
    E_coeff.x.array[:] = penalised_E(rho_fn.x.array, penal)

    T_vec, ds, tag = traction_load

    mu_eff    = E_coeff * mu0
    lmbda_eff = E_coeff * lmbda0

    a = ufl.inner(
        lmbda_eff * ufl.tr(epsilon(u)) * ufl.Identity(dim) + 2.0 * mu_eff * epsilon(u),
        epsilon(v)
    ) * ufl.dx

    L = ufl.inner(T_vec, v) * ds(tag)

    problem = LinearProblem(
        a, L, bcs=bcs,
        petsc_options={"ksp_type": "cg", "pc_type": "hypre"},
    )
    uh = problem.solve()
    return uh


def compute_sensitivity(domain, Q, rho_fn, uh, penal, mu0, lmbda0):
    dim = domain.geometry.dim
    rho = rho_fn.x.array

    sig0 = sigma_expr(uh, mu0, lmbda0, dim)
    sed  = ufl.inner(sig0, epsilon(uh))

    # Assemble element integrals directly (No projection solve!)
    v_w = ufl.TestFunction(Q)
    L_ce = fem.form(sed * v_w * ufl.dx)
    
    ce_vec = assemble_vector(L_ce)        # ← fem_petsc version
    ce_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    ce = ce_vec.array.copy()

    # ∂C/∂ρ_e = −p · ρ^{p−1} · ce
    dc = -penal * (rho ** (penal - 1)) * ce

    return dc, ce

def oc_update(rho: np.ndarray, dc: np.ndarray, volfrac: float,
              move: float = 0.05) -> np.ndarray:
    """
    Optimality Criteria density update with bisection on the Lagrange multiplier.

    ρ_new = clip( ρ · sqrt(−∂C/∂ρ / λ),  ρ−move,  ρ+move,  [1e-3, 1] )
    Volume constraint: mean(ρ_new) = volfrac
    """
    l1, l2 = 0.0, 1e9
    while (l2 - l1) / (l1 + l2 + 1e-40) > 1e-4:
        lmid    = 0.5 * (l1 + l2)
        rho_new = np.clip(
            rho * np.sqrt(np.maximum(-dc / (lmid + 1e-40), 0.0)),
            np.maximum(rho - move, 1e-3),
            np.minimum(rho + move, 1.0),
        )
        if rho_new.mean() > volfrac:
            l1 = lmid
        else:
            l2 = lmid
    return rho_new


def oc_update_masked(rho, dc, volfrac, void_mask=None, move=0.05):
    if void_mask is None:
        return oc_update(rho, dc, volfrac, move)
    
    active = ~void_mask
    rho_active_new = oc_update(rho[active], dc[active], volfrac, move)
    
    rho_new = rho.copy()
    rho_new[active] = rho_active_new
    rho_new[void_mask] = E_MIN
    return rho_new


# ══════════════════════════════════════════════════════════════════════════════
# MILESTONE 3 — Density filter + convergence metrics
# ══════════════════════════════════════════════════════════════════════════════

def build_helmholtz_filter(Q, r_min: float):
    domain = Q.mesh
    # 1. Create a continuous space for the PDE
    F = fem.functionspace(domain, ("Lagrange", 1))
    
    rh = fem.Constant(domain, default_scalar_type(r_min**2))

    rho_trial = ufl.TrialFunction(F)
    rho_test  = ufl.TestFunction(F)
    dx        = ufl.Measure("dx", domain=domain)

    # 2. Assemble continuous diffusion matrix
    a_f = (rh * ufl.dot(ufl.grad(rho_trial), ufl.grad(rho_test)) + 
           rho_trial * rho_test) * dx

    A_f = fem_petsc.assemble_matrix(fem.form(a_f))
    A_f.assemble()

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A_f)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.setUp()

    return solver, F, rho_test, dx # Return F instead of Q


def apply_helmholtz_filter(rho_fn: fem.Function, solver, F, rho_test, dx) -> fem.Function:
    L_f = rho_fn * rho_test * dx
    b = fem_petsc.assemble_vector(fem.form(L_f))
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    rho_cg = fem.Function(F)
    solver.solve(b, rho_cg.x.petsc_vec)   # ← .vector → .x.petsc_vec
    rho_cg.x.scatter_forward()

    Q = rho_fn.function_space
    rho_filtered = fem.Function(Q)
    expr = fem.Expression(rho_cg, Q.element.interpolation_points())
    rho_filtered.interpolate(expr)
    
    return rho_filtered
def apply_bcs_from_spec(V, domain, spec: ProblemSpec):
    bcs = []
    for bc_data in spec.bcs:
        if bc_data.location == "left_edge":
            def boundary(x): return np.isclose(x[0], 0.0)
        elif bc_data.location == "bottom_edge":
            def boundary(x): return np.isclose(x[1], 0.0)
        # ... add more mappings ...
        
        dofs = fem.locate_dofs_geometrical(V, boundary)
        bc = fem.dirichletbc(default_scalar_type(bc_data.value), dofs, V)
        bcs.append(bc)
    return bcs

class ConvergenceMetrics:
    """Lightweight container that logs per-iteration metrics."""

    def __init__(self):
        self.data: list[dict] = []

    def record(self, itr: int, compliance: float, vol_frac: float,
               l2_delta: float, penal: float):
        self.data.append({
            "iter":       itr,
            "compliance": compliance,
            "vol_frac":   vol_frac,
            "l2_delta":   l2_delta,
            "penal":      penal,
        })
        print(
            f"Iter {itr:4d} | p={penal:.2f} | "
            f"C={compliance:10.4f} | vol={vol_frac:.4f} | "
            f"Δρ(L2)={l2_delta:.2e}"
        )

    def arrays(self):
        keys = ["iter", "compliance", "vol_frac", "l2_delta", "penal"]
        return {k: np.array([d[k] for d in self.data]) for k in keys}


# ══════════════════════════════════════════════════════════════════════════════
# MILESTONE 4 — Density field plotting
# ══════════════════════════════════════════════════════════════════════════════

def _rho_to_image(rho_vals: np.ndarray, nelx: int, nely: int,
                  domain, Lx: float, Ly: float) -> np.ndarray:
    """Map DG0 array to 2D image using actual cell midpoint coordinates."""
    num_local = domain.topology.index_map(domain.topology.dim).size_local
    midpoints = mesh.compute_midpoints(
        domain, domain.topology.dim,
        np.arange(num_local, dtype=np.int32)
    )
    dx, dy = Lx / nelx, Ly / nely
    ix = np.clip(np.floor(midpoints[:, 0] / dx).astype(int), 0, nelx - 1)
    iy = np.clip(np.floor(midpoints[:, 1] / dy).astype(int), 0, nely - 1)

    img = np.zeros((nely, nelx))
    img[iy, ix] = rho_vals[:num_local]
    return img


def plot_density(rho_vals, nelx, nely, itr, out_dir, geometry,
                 domain, Lx, Ly, compliance=None):        # ← add domain, Lx, Ly
    img = _rho_to_image(rho_vals, nelx, nely, domain, Lx, Ly)

    fig, ax = plt.subplots(figsize=(max(6, nelx // 10), max(2, nely // 10)))
    ax.imshow(
        1.0 - img,           # invert: solid=black, void=white
        cmap="gray",
        origin="lower",
        aspect="equal",
        vmin=0, vmax=1,
    )
    title = f"{geometry}  iter={itr}"
    if compliance is not None:
        title += f"  C={compliance:.4f}"
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.tight_layout(pad=0.3)

    fname = os.path.join(out_dir, f"{geometry}_rho_{itr:04d}.png")
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_summary(metrics: ConvergenceMetrics, rho_vals: np.ndarray,
                 nelx: int, nely: int, geometry: str, out_dir: str,
                 domain, Lx: float, Ly: float):          # ← add three params
    m   = metrics.arrays()
    img = _rho_to_image(rho_vals, nelx, nely, domain, Lx, Ly)

    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    fig.suptitle(f"SIMP — {geometry}", fontsize=13)

    # 1. Final topology
    axes[0, 0].imshow(1 - img, cmap="gray", origin="lower", aspect="equal")
    axes[0, 0].set_title("Final Topology")
    axes[0, 0].axis("off")

    # 2. Density histogram
    axes[0, 1].hist(rho_vals, bins=60, color="steelblue",
                    edgecolor="none", linewidth=0)
    axes[0, 1].set_xlabel("Density ρ")
    axes[0, 1].set_ylabel("Elements")
    axes[0, 1].set_title("Density Distribution")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Penalisation schedule
    axes[0, 2].plot(m["iter"], m["penal"], color="seagreen", lw=1.5)
    axes[0, 2].set_xlabel("Iteration")
    axes[0, 2].set_ylabel("p")
    axes[0, 2].set_title("Penalisation Schedule")
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Compliance history
    axes[1, 0].plot(m["iter"], m["compliance"], color="steelblue", lw=1.5)
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Compliance C")
    axes[1, 0].set_title("Compliance History")
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Volume fraction history
    axes[1, 1].plot(m["iter"], m["vol_frac"], color="darkorange", lw=1.5)
    axes[1, 1].axhline(rho_vals.mean(), color="red", ls="--", lw=1,
                       label=f"target ≈ {rho_vals.mean():.3f}")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Volume fraction")
    axes[1, 1].set_title("Volume Fraction")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    # 6. L2 delta (convergence)
    axes[1, 2].semilogy(m["iter"], m["l2_delta"], color="purple", lw=1.5)
    axes[1, 2].set_xlabel("Iteration")
    axes[1, 2].set_ylabel("‖Δρ‖₂")
    axes[1, 2].set_title("L2 Density Change")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, f"{geometry}_summary.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Summary plot → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Top-level SIMP solver
# ══════════════════════════════════════════════════════════════════════════════
from dolfinx.io import XDMFFile

class XDMFExporter:
    """
    Writes the density field at each iteration to a single XDMF+H5 pair
    that ParaView can open as a time series.

    Usage inside SIMPSolver.optimize():
        exporter = XDMFExporter(self.domain, self.Q, self.output_dir, self.geometry)
        # inside loop:
        exporter.write(self.rho_fn, itr)
        # after loop:
        exporter.close()
    """

    def __init__(self, domain, Q, output_dir: str, geometry: str):
        self.path = os.path.join(output_dir, f"{geometry}_density.xdmf")
        self.file = XDMFFile(domain.comm, self.path, "w")
        self.file.write_mesh(domain)

    def write(self, rho_fn: fem.Function, itr: int):
        rho_fn.name = "density"
        self.file.write_function(rho_fn, float(itr))

    def close(self):
        self.file.close()
        print(f"XDMF exported → {self.path}")
class SIMPSolver:
    """
    Full DOLFINx SIMP solver.

    Parameters
    ----------
    nelx, nely      : mesh divisions (x, y)
    Lx, Ly          : domain dimensions
    volfrac         : target volume fraction
    penal_start/end : SIMP penalisation continuation (1→3)
    penal_steps     : iterations over which p ramps
    r_min           : Helmholtz filter radius
    max_iter        : iteration cap
    change_tol      : L2(Δρ) convergence threshold
    geometry        : "cantilever" | "lbracket"
    plot_every      : save density image every N iters (0 = never)
    output_dir      : directory for all output files
    """

    def __init__(
        self,
        spec: ProblemSpec,                # The single source of truth
        Lx: float = 1.0, 
        Ly: float = 1.0,
        penal_steps: int = 40,
        max_iter: int = 300,
        change_tol: float = 5e-3,
        plot_every: int = 10,
        output_dir: str = "results",
    ):
        # 1. Extract from spec
        self.spec = spec
        self.nelx = spec.mesh.nx
        self.nely = spec.mesh.ny
        self.volfrac = spec.simp.vol_frac
        self.penal_end = spec.simp.penal
        self.r_min = spec.simp.r_min
        self.geometry = spec.name.lower().replace(" ", "_")

        # 2. Assign static/solver defaults
        self.Lx = Lx
        self.Ly = Ly
        self.penal_start = 1.0
        self.penal_steps = penal_steps
        self.max_iter = max_iter
        self.change_tol = change_tol
        self.plot_every = plot_every
        self.output_dir = output_dir
        
        # 3. State & Config
        self.rho_frames = []
        self.config = {
            "penal": self.penal_end,
            "r_min": self.r_min,
            "volfrac_target": self.volfrac,
        }
        os.makedirs(output_dir, exist_ok=True)

        # ── Mesh & spaces ──────────────────────────────────────────────────
        self.domain = build_mesh(self.nelx, self.nely, Lx, Ly)
        self.V, self.Q = build_function_spaces(self.domain)

        # ── Material ───────────────────────────────────────────────────────
        self.mu0, self.lmbda0 = material_constants(E=1.0, nu=0.3)

        # ── BCs & traction ─────────────────────────────────────────────────
        fdim = self.domain.topology.dim - 1

        if self.geometry == "cantilever":
            self.bcs = build_bcs_cantilever(self.V, self.domain, Lx)
            # Traction on the entire right edge
            ds = build_traction_measure(
                self.domain, fdim,
                lambda x: np.isclose(x[0], Lx),
                tag=1,
            )
            T  = fem.Constant(
                self.domain,
                default_scalar_type([0.0, -1.0]),
            )
            self.traction_load = (T, ds, 1)

        elif self.geometry == "lbracket":
            self.bcs = build_bcs_lbracket(self.V, self.domain, Lx, Ly)
            # Point-like load at right edge, mid-height
            def right_mid(x):
                return np.logical_and(
                    np.isclose(x[0], Lx),
                    x[1] <= Ly / 2.0 + Ly / (2 * self.nely),
                )
            ds = build_traction_measure(self.domain, fdim, right_mid, tag=2)
            T  = fem.Constant(
                self.domain,
                default_scalar_type([0.0, -1.0]),
            )
            self.traction_load = (T, ds, 2)
        else:
            raise ValueError(f"Unknown geometry: {self.geometry!r}")

        # ── Density field (DG0) ────────────────────────────────────────────
        self.rho_fn = fem.Function(self.Q)
        self.rho_fn.x.array[:] = self.volfrac

        # L-bracket: zero out top-right quadrant
        self.void_mask = None
        if self.geometry == "lbracket":
            self.void_mask = self._lbracket_void_mask()
            self.rho_fn.x.array[self.void_mask] = E_MIN
            active = ~self.void_mask
            self.rho_fn.x.array[active] = self.volfrac

        # ── Helmholtz filter ───────────────────────────────────────────────
        (self.filter_solver,
        self.filter_space,
        self.filter_test,
        self.filter_dx) = build_helmholtz_filter(self.Q, self.r_min)

        # ── Metrics ────────────────────────────────────────────────────────
        self.metrics = ConvergenceMetrics()

    # ── Helpers ────────────────────────────────────────────────────────────

    def _lbracket_void_mask(self) -> np.ndarray:
        num_local = self.domain.topology.index_map(self.domain.topology.dim).size_local
        midpoints = mesh.compute_midpoints(
            self.domain,
            self.domain.topology.dim,
            np.arange(num_local, dtype=np.int32),
        )
        return (midpoints[:, 0] >= self.Lx / 2.0) & (midpoints[:, 1] >= self.Ly / 2.0)

    def _penal(self, itr: int) -> float:
    # Use steered penal if past ramp, otherwise use schedule
        if itr >= self.penal_steps:
            return self.config.get("penal", self.penal_end)
        t = min(itr / max(self.penal_steps, 1), 1.0)
        return self.penal_start + t * (self.penal_end - self.penal_start)

    # ── Core optimisation loop ─────────────────────────────────────────────

    def optimize(self):
        from agents import PhysicsGate, SteeringAgent, CriticAgent, build_metrics_snapshot

        gate    = PhysicsGate(volfrac_target=self.volfrac)
        steerer = SteeringAgent(config=self.config, every_n=10)
        l2_delta = np.inf
        exporter = XDMFExporter(self.domain, self.Q, self.output_dir, self.geometry)

        for itr in range(self.max_iter):
            penal = self._penal(itr)

            # --- Filter the physical density before solving ----------------
            rho_phys = apply_helmholtz_filter(
                self.rho_fn,
                self.filter_solver,
                self.filter_space,
                self.filter_test,
                self.filter_dx,
            )
            # --- FEA -------------------------------------------------------
            uh = solve_elasticity(
                self.domain, self.V, self.bcs,
                rho_phys, penal,
                self.mu0, self.lmbda0,
                self.traction_load,
            )

            # --- Sensitivity -----------------------------------------------
            dc, ce = compute_sensitivity(
                self.domain, self.Q, rho_phys,
                uh, penal, self.mu0, self.lmbda0,
            )

            # --- Filter sensitivity (chain rule through Helmholtz filter) --
            # Wrap dc in a fem.Function and filter it
            dc_fn = fem.Function(self.Q)
            dc_fn.x.array[:] = dc
            dc_filtered = apply_helmholtz_filter(
                dc_fn, self.filter_solver, self.filter_space,
                self.filter_test, self.filter_dx,
            )
            dc = dc_filtered.x.array.copy()

            # --- OC update -------------------------------------------------
            rho_old = self.rho_fn.x.array.copy()
            
            # Use the live config dictionary instead of the static self.volfrac!
            current_volfrac = self.config.get("volfrac_target", self.volfrac)

            rho_new = oc_update_masked(
                rho_old, dc, current_volfrac, self.void_mask, # <-- FIXED
            )
            self.rho_fn.x.array[:] = rho_new
            self.rho_fn.x.scatter_forward()
            self.rho_frames.append(self.rho_fn.x.array.copy())
            exporter.write(self.rho_fn, itr)
            rho = self.rho_fn.x.array
            print(f"Iter {itr:3d} | vol={rho.mean():.4f}, min={rho.min():.3f}, max={rho.max():.3f}")
            # --- Metrics ---------------------------------------------------
            rho = rho_phys.x.array
            compliance = float(np.dot(penalised_E(rho, penal), ce))
            vol_frac   = float(rho_new.mean())
            l2_delta   = float(np.linalg.norm(rho_new - rho_old))

            self.metrics.record(itr, compliance, vol_frac, l2_delta, penal)

            # Physics gate — warns but doesn't crash during ramp phase
            passed, reasons = gate.check(
                rho_new, self.nelx, self.nely,
                [d["compliance"] for d in self.metrics.data],
                self.void_mask,
                itr=itr,                         # <-- NEW
                penal_steps=self.penal_steps     # <-- NEW
            )
            if not passed:
                print("[PhysicsGate] " + " | ".join(r for r in reasons if "FAIL" in r))

            # Steering agent — calls Claude every 10 iters
            snapshot = build_metrics_snapshot(
                itr, 
                [d["compliance"] for d in self.metrics.data],
                [d["l2_delta"] for d in self.metrics.data],
                [d["penal"] for d in self.metrics.data],      # <-- NEW ARGUMENT
                rho_new, self.void_mask, self.nelx, self.nely,
                self.domain, self.Lx, self.Ly
            )
            steerer.maybe_steer(itr, snapshot)
            new_r = self.config.get("r_min", self.r_min)
            if abs(new_r - self.r_min) > 1e-6:
                self.r_min = new_r
                (self.filter_solver,
                self.filter_space,
                self.filter_test,
                self.filter_dx) = build_helmholtz_filter(self.Q, self.r_min)
                print(f"  [Filter rebuilt] r_min={self.r_min:.3f}")
            # --- Density plot ----------------------------------------------
            if self.plot_every > 0 and itr % self.plot_every == 0:
                plot_density(
                    rho_new, self.nelx, self.nely,
                    itr, self.output_dir, self.geometry,
                    self.domain, self.Lx, self.Ly,      # ← add these
                    compliance=compliance,
                )
            
            # --- Convergence check -----------------------------------------
            if itr >= self.penal_steps and l2_delta < self.change_tol:
                print(f"\nConverged at iteration {itr}  (‖Δρ‖₂ = {l2_delta:.2e})")
                break

        else:
            print(
                f"\nWARNING: did not converge in {self.max_iter} iterations. "
                f"Final ‖Δρ‖₂ = {l2_delta:.2e}"
            )
        # Final density frame
        plot_density(
            self.rho_fn.x.array, self.nelx, self.nely,
            itr, self.output_dir, self.geometry,
            self.domain, self.Lx, self.Ly,
        )
        plot_summary(
            self.metrics, self.rho_fn.x.array,
            self.nelx, self.nely, self.geometry, self.output_dir,
            self.domain, self.Lx, self.Ly,    # ← add these
        )
        self.save_gif()
        critic  = CriticAgent()
        metrics_dict = {"compliance": [d["compliance"] for d in self.metrics.data],
                        "l2_delta":   [d["l2_delta"]   for d in self.metrics.data]}
        sanity_dict  = {"converged": l2_delta < self.change_tol,
                        "physics_gate": {"last_passed": passed}}
        assessment = critic.critique(metrics_dict, sanity_dict,
                                    self.rho_fn.x.array, self.nelx, self.nely)
        print("\n─── Critic Assessment ──────────────────────────────────")
        print(assessment)
        print("────────────────────────────────────────────────────────\n")
        self._sanity_checks()
        exporter.close()

    # ── Post-processing ────────────────────────────────────────────────────

    def _sanity_checks(self):
        rho = self.rho_fn.x.array
        print("\n─── Sanity Checks ──────────────────────────────────────")

        active = rho if self.void_mask is None else rho[~self.void_mask]
        vol_err = abs(active.mean() - self.volfrac)
        tag = "OK  " if vol_err < 0.01 else "WARN"
        print(f"[{tag}] Volume fraction: target={self.volfrac:.3f}  "
              f"actual={active.mean():.4f}")

        frac_solid = np.mean(rho > 0.9)
        frac_void  = np.mean(rho < 0.1)
        frac_grey  = 1.0 - frac_solid - frac_void
        tag = "OK  " if frac_grey < 0.05 else "WARN"
        print(f"[{tag}] Binarisation: solid={frac_solid:.1%}  "
              f"void={frac_void:.1%}  grey={frac_grey:.1%}")

        if len(self.metrics.data) >= 10:
            last   = [d["compliance"] for d in self.metrics.data[-10:]]
            spread = (max(last) - min(last)) / (np.mean(last) + 1e-30)
            tag    = "OK  " if spread < 0.005 else "WARN"
            print(f"[{tag}] Compliance spread (last 10 iters): {spread:.3%}")

        print("────────────────────────────────────────────────────────\n")

    def save_gif(self, fps=10, every=2):
        frames_to_use = self.rho_frames[::every]
        images = []

        for rho in frames_to_use:
            img = _rho_to_image(rho, self.nelx, self.nely,
                                self.domain, self.Lx, self.Ly)
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.imshow(1 - img, cmap="gray", aspect="auto", vmin=0, vmax=1)
            ax.axis("off")
            fig.tight_layout(pad=0)

            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf  = fig.canvas.buffer_rgba()
            img  = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))[:, :, :3]
            images.append(img)
            plt.close(fig)

        out = os.path.join(self.output_dir, f"{self.geometry}_evolution.gif")
        imageio.mimsave(out, images, fps=fps)
        print(f"Saved GIF  → {out}  ({len(images)} frames @ {fps}fps)")
# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="DOLFINx SIMP topology optimisation")
    p.add_argument("--geometry",    default="cantilever", choices=["cantilever", "lbracket"])
    p.add_argument("--nelx",        type=int,   default=60)
    p.add_argument("--nely",        type=int,   default=20)
    p.add_argument("--Lx",          type=float, default=1.0)
    p.add_argument("--Ly",          type=float, default=1.0)
    p.add_argument("--volfrac",     type=float, default=0.4)
    p.add_argument("--penal-start", type=float, default=1.0,  dest="penal_start")
    p.add_argument("--penal-end",   type=float, default=3.0,  dest="penal_end")
    p.add_argument("--penal-steps", type=int,   default=40,   dest="penal_steps")
    p.add_argument("--rmin",        type=float, default=0.04)
    p.add_argument("--max-iter",    type=int,   default=300,  dest="max_iter")
    p.add_argument("--tol",         type=float, default=5e-3)
    p.add_argument("--plot-every",  type=int,   default=10,   dest="plot_every")
    p.add_argument("--output-dir",  default="results",        dest="output_dir")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    solver = SIMPSolver(
        nelx        = args.nelx,
        nely        = args.nely,
        Lx          = args.Lx,
        Ly          = args.Ly,
        volfrac     = args.volfrac,
        penal_start = args.penal_start,
        penal_end   = args.penal_end,
        penal_steps = args.penal_steps,
        r_min       = args.rmin,
        max_iter    = args.max_iter,
        change_tol  = args.tol,
        geometry    = args.geometry,
        plot_every  = args.plot_every,
        output_dir  = args.output_dir,
    )
    solver.optimize()
"""SIMP/MMA driver for the Holmberg stress-constrained formulations.

Implements formulations P1, P2 and P3 of Holmberg et al. (2013) Sec. 2 and the
adjoint sensitivity analysis of Sec. 7.

One evaluator, two analysis models
----------------------------------
``Evaluator`` is parameterized by ``spec.geometry.model`` rather than duplicated:

* ``plane_stress``  -- Holmberg's 2-D Q4 model, 2 dof/node, 3 stress components
* ``extruded_3d``   -- the 2.5D extrusion of ``solid3d.py``, 3 dof/node, 6 stress
                       components, density constant through the thickness

Everything downstream -- objective, clustering, adjoint, MMA loop -- is written
against an abstract system interface, so the two models cannot drift apart and a
result from one is comparable with the other cell by cell.  Design variables are
always **per in-plane cell**; ``design_of_elem`` maps elements to them (the
identity for the 2-D model).

Multiple load cases are supported.  Each case contributes its own clustered
stress constraints, so the constraint count is ``n_cases * n_clusters``, and the
compliance objective is the sum over cases.  A case the model cannot represent
raises ``ModelCannotRepresent`` rather than returning a misleading number.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from project.topopt.stress import solid3d
from project.topopt.stress.clustering import assign_clusters
from project.topopt.stress.domain import LBracketGeometry, build_lbracket
from project.topopt.stress.fem_q4 import Q4System
from project.topopt.stress.filters import cone_filter_matrix
from project.topopt.stress.mma_multi import MultiConstraintMMA
from project.topopt.stress.problem import ModelCannotRepresent, StressSpec
from project.topopt.stress.stress import d_eta_S, d_von_mises, eta_S, von_mises

IN_PLANE_AXIS = {"x": 0, "y": 1}


def build_force(mesh, magnitude: float, distribute_nodes: int,
                direction: str = "y") -> np.ndarray:
    """Tip load on the 2-D mesh, spread over ``distribute_nodes`` right-edge nodes.

    Raises ``ModelCannotRepresent`` for an out-of-plane direction: in plane stress
    each node carries only ``u, v``, so a z load has no degree of freedom to act
    on.  See Holmberg Sec. 8.1/9.4 for the distribution option.
    """
    if direction == "z":
        raise ModelCannotRepresent(
            "a plane-stress model has no out-of-plane degree of freedom, so a load "
            "along z cannot be applied at all (each node carries only u, v). Promote "
            "the analysis model to 'extruded_3d' to make this load case expressible."
        )
    if direction not in IN_PLANE_AXIS:
        raise ValueError(f"unknown load direction {direction!r}; use x, y or z")
    axis = IN_PLANE_AXIS[direction]
    tol = 1.0e-9
    arm = mesh.geom.arm
    on_edge = np.isclose(mesh.coords[:, 0], mesh.geom.L, atol=tol) & (mesh.coords[:, 1] <= arm + tol)
    nodes = np.flatnonzero(on_edge)
    nodes = nodes[np.argsort(-mesh.coords[nodes, 1])]
    k = max(1, int(distribute_nodes))
    if k > nodes.size:
        raise ValueError(f"cannot distribute over {k} nodes; right edge has {nodes.size}")
    chosen = nodes[:k]
    F = np.zeros(mesh.n_dof, dtype=float)
    F[2 * chosen + axis] = -float(magnitude) / k
    return F


class Evaluator:
    """Analysis, objective, constraints and sensitivities for one specification."""

    def __init__(self, spec: StressSpec):
        self.spec = spec
        g, m, cs, op = spec.geometry, spec.material, spec.constraints, spec.optimizer
        self.model = g.model

        if self.model == "plane_stress":
            geom = LBracketGeometry(
                L=g.L, arm_fraction=g.arm_fraction, thickness=g.thickness,
                n_cells_per_side=g.n_cells_per_side,
                exclude_nx=g.exclude_nx, exclude_ny=g.exclude_ny,
            )
            self.mesh = build_lbracket(
                geom, corner_fillet_radius=g.corner_fillet_radius,
                exclude_load_elements=g.exclude_load_elements,
            )
            self.system = Q4System(self.mesh, E=m.E, nu=m.nu)
            self.inplane_centroids = self.mesh.centroids
            self.design_of_elem = np.arange(self.mesh.n_elem)
            self.designable = self.mesh.designable
            self.centroids2d = self.mesh.centroids
            self.vm, self.dvm = von_mises, d_von_mises
            self.n_layers = 1
            self._force = lambda lc: build_force(
                self.mesh, lc.magnitude, lc.distribute_nodes, lc.direction)
            self.h = geom.h
            self.corner_xy = self.mesh.reentrant_corner()
            self.load_xy = self.mesh.load_point()
        elif self.model == "extruded_3d":
            self.mesh = solid3d.build_lbracket_3d(
                L=g.L, arm_fraction=g.arm_fraction, thickness=g.thickness,
                n_cells_per_side=g.n_cells_per_side, n_layers=g.n_layers,
                corner_fillet_radius=g.corner_fillet_radius,
                exclude_load_elements=g.exclude_load_elements,
                exclude_nx=g.exclude_nx, exclude_ny=g.exclude_ny,
            )
            self.system = solid3d.H8System(self.mesh, E=m.E, nu=m.nu)
            self.inplane_centroids = self.mesh.inplane_centroids
            self.design_of_elem = self.mesh.design_of_elem
            self.designable = self.mesh.designable
            self.centroids2d = self.mesh.centroids[:, :2]
            self.vm, self.dvm = solid3d.von_mises_3d, solid3d.d_von_mises_3d
            self.n_layers = self.mesh.n_layers
            self._force = lambda lc: solid3d.build_force_3d(
                self.mesh, lc.magnitude, lc.distribute_nodes, lc.direction)
            self.h = self.mesh.h
            self.corner_xy = self.mesh.reentrant_corner()
            self.load_xy = self.mesh.load_point()
        else:
            raise ValueError(f"unknown analysis model {self.model!r}")

        self.n_elem = self.mesh.n_elem
        self.n_inplane = self.inplane_centroids.shape[0]
        self.W = cone_filter_matrix(self.inplane_centroids, op.r0_elements * self.h)
        self.Wt = self.W.T.tocsr()
        self.forces = [self._force(lc) for lc in spec.load_cases]
        self.case_names = [lc.name or f"case{i}" for i, lc in enumerate(spec.load_cases)]
        self.design_idx = np.flatnonzero(self.designable)
        self._edof_flat = self.mesh.edof.ravel()

        # Mass depends only on the in-plane density: the extrusion multiplies the
        # element count and divides the element volume by the same n_layers.
        self.m_cell = m.density * self.h * self.h * g.thickness
        self.mass_full = self.m_cell * self.n_inplane
        self.mass_limit = cs.mass_fraction_limit * self.mass_full

        self.uses_stress = spec.formulation in ("P1", "P2")
        self.uses_mass = spec.formulation in ("P2", "P3")
        if not (self.uses_stress or self.uses_mass):
            raise ValueError(f"formulation {spec.formulation!r} has no constraints")
        self.n_cases = len(self.forces)
        self.n_con = (cs.n_clusters * self.n_cases if self.uses_stress else 0) \
            + (1 if self.uses_mass else 0)
        self.C0 = None

    # -- helpers --------------------------------------------------------------

    def elem_density(self, rho: np.ndarray) -> np.ndarray:
        return rho[self.design_of_elem]

    def to_inplane(self, per_elem: np.ndarray) -> np.ndarray:
        return np.bincount(self.design_of_elem, weights=per_elem, minlength=self.n_inplane)

    def initial_design(self) -> np.ndarray:
        op, cs = self.spec.optimizer, self.spec.constraints
        x = np.ones(self.n_inplane, dtype=float)
        x0 = op.init_density if self.spec.formulation == "P1" else cs.mass_fraction_limit
        x[self.design_idx] = np.clip(x0, op.x_min, 1.0)
        return x

    def state(self, x_full: np.ndarray) -> dict:
        op = self.spec.optimizer
        q = op.penal
        rho = np.clip(self.W @ x_full, op.x_min, 1.0)
        rho_e = self.elem_density(rho)
        self.system.factorize(rho_e ** q)
        cases = []
        for F in self.forces:
            u = self.system.solve(F)
            sigma = self.system.solid_stress(u)
            vmhat = self.vm(sigma)
            cases.append({"u": u, "sigma": sigma, "vmhat": vmhat,
                          "svm": eta_S(rho_e) * vmhat,
                          "compliance": 0.5 * float(F @ u)})
        return {"rho": rho, "rho_e": rho_e, "cases": cases,
                "compliance": float(sum(c["compliance"] for c in cases)),
                "mass": self.m_cell * float(rho.sum())}

    def evaluate(self, x_full: np.ndarray, clusters=None) -> dict:
        cs, op = self.spec.constraints, self.spec.optimizer
        q, p, sbar = op.penal, cs.p_norm, cs.stress_limit
        st = self.state(x_full)
        rho, rho_e = st["rho"], st["rho_e"]

        if self.C0 is None:
            self.C0 = max(st["compliance"], 1.0e-30)

        if self.spec.formulation == "P1":
            f0 = st["mass"] / self.mass_full
            df0_dinp = np.full(self.n_inplane, self.m_cell / self.mass_full)
        else:
            acc = np.zeros(self.n_elem)
            for c in st["cases"]:
                acc += -0.5 * q * rho_e ** (q - 1.0) * \
                    self.system.element_strain_energy_density(c["u"])
            f0 = st["compliance"] / self.C0
            df0_dinp = self.to_inplane(acc) / self.C0

        fval, dfd_inp = [], []

        if self.uses_stress:
            if clusters is None:
                clusters = [assign_clusters(c["svm"], cs.n_clusters, cs.clustering)
                            for c in st["cases"]]
            etaS, detaS = eta_S(rho_e), d_eta_S(rho_e)
            for ci, c in enumerate(st["cases"]):
                svm, u = c["svm"], c["u"]
                nhat = self.dvm(c["sigma"], c["vmhat"])
                u_e = u[self.mesh.edof]
                for grp in clusters[ci]:
                    Ni = grp.size
                    G = float(np.mean((svm[grp] / sbar) ** p)) ** (1.0 / p)
                    fval.append(G - 1.0)
                    coef = np.zeros(self.n_elem)
                    coef[grp] = (G ** (1.0 - p)) * (svm[grp] ** (p - 1.0)) / (Ni * sbar ** p)
                    dG = coef * detaS * c["vmhat"]                         # explicit
                    contrib = (nhat * (coef * etaS)[:, None]) @ self.system.DB
                    fadj = np.bincount(self._edof_flat, weights=contrib.ravel(),
                                       minlength=self.mesh.n_dof)
                    lam_e = self.system.solve(fadj)[self.mesh.edof]
                    dG -= q * rho_e ** (q - 1.0) * np.einsum(
                        "ei,ij,ej->e", lam_e, self.system.ke, u_e)          # implicit
                    dfd_inp.append(self.to_inplane(dG))

        if self.uses_mass:
            fval.append(st["mass"] / self.mass_limit - 1.0)
            dfd_inp.append(np.full(self.n_inplane, self.m_cell / self.mass_limit))

        return {
            **st,
            "f0": float(f0),
            "df0dx": (self.Wt @ df0_dinp)[self.design_idx],
            "fval": np.asarray(fval, dtype=float),
            "dfdx": np.stack([(self.Wt @ d)[self.design_idx] for d in dfd_inp], axis=0),
            "clusters": clusters,
        }


@dataclass
class RunResult:
    spec: StressSpec
    mesh: object
    rho: np.ndarray                    # per in-plane cell
    rho_elem: np.ndarray               # per element
    sigma_vm: np.ndarray               # per element, envelope over load cases
    sigma_vm_by_case: dict
    centroids2d: np.ndarray
    corner_xy: np.ndarray
    load_xy: np.ndarray
    model: str
    n_layers: int
    compliance: float
    mass: float
    mass_fraction: float
    max_stress: float
    max_stress_ratio: float
    iterations: int
    converged: bool
    termination: str
    stress_feasible: bool
    max_constraint: float
    history: dict = field(default_factory=dict)
    wall_seconds: float = 0.0

    def summary(self) -> dict:
        return {
            "formulation": self.spec.formulation,
            "model": self.model,
            "n_layers": self.n_layers,
            "load_cases": [f"{lc.name}:{lc.direction}" for lc in self.spec.load_cases],
            "iterations": self.iterations,
            "converged": self.converged,
            "termination": self.termination,
            "compliance_N_mm": self.compliance,
            "mass_kg": self.mass * 1000.0,
            "mass_fraction": self.mass_fraction,
            "max_von_mises_MPa": self.max_stress,
            "max_von_mises_by_case": {k: float(v.max()) for k, v in self.sigma_vm_by_case.items()},
            "max_stress_ratio": self.max_stress_ratio,
            "stress_feasible": self.stress_feasible,
            "max_constraint_value": self.max_constraint,
            "wall_seconds": self.wall_seconds,
        }


def solve_stress_problem(spec: StressSpec, *, record_history: bool = True) -> RunResult:
    t0 = time.time()
    cs, op = spec.constraints, spec.optimizer
    ev = Evaluator(spec)
    x_full = ev.initial_design()
    mma = MultiConstraintMMA(ev.design_idx.size, ev.n_con, x_min=op.x_min, move=op.move)

    clusters = None
    change = np.inf
    converged = False
    termination = "max_iter_reached"
    hist = {k: [] for k in ("objective", "max_stress", "mass_fraction", "change", "max_constraint")}

    for it in range(1, op.max_iter + 1):
        recluster = (
            ev.uses_stress
            and (clusters is None or (cs.recluster_every > 0 and (it - 1) % cs.recluster_every == 0))
        )
        out = ev.evaluate(x_full, clusters=None if recluster else clusters)
        clusters = out["clusters"]
        max_con = float(out["fval"].max())
        peak = max(float(c["svm"].max()) for c in out["cases"])

        if record_history:
            hist["objective"].append(out["f0"])
            hist["max_stress"].append(peak)
            hist["mass_fraction"].append(float(out["mass"] / ev.mass_full))
            hist["change"].append(float(change) if np.isfinite(change) else None)
            hist["max_constraint"].append(max_con)

        x_new = np.clip(
            mma.update(x_full[ev.design_idx], out["f0"], out["df0dx"], out["fval"], out["dfdx"]),
            op.x_min, 1.0,
        )
        change = float(np.max(np.abs(x_new - x_full[ev.design_idx])))
        x_full[ev.design_idx] = x_new

        if change < op.tol_change and max_con <= op.feasibility_tol:
            converged, termination = True, "design_converged_and_feasible"
            break

    final = ev.evaluate(x_full, clusters=None)
    max_con = float(final["fval"].max())
    by_case = {n: c["svm"] for n, c in zip(ev.case_names, final["cases"])}
    envelope = np.max(np.stack(list(by_case.values()), axis=0), axis=0)
    return RunResult(
        spec=spec, mesh=ev.mesh, rho=final["rho"], rho_elem=final["rho_e"],
        sigma_vm=envelope, sigma_vm_by_case=by_case,
        centroids2d=ev.centroids2d, corner_xy=ev.corner_xy, load_xy=ev.load_xy,
        model=ev.model, n_layers=ev.n_layers,
        compliance=final["compliance"], mass=final["mass"],
        mass_fraction=final["mass"] / ev.mass_full,
        max_stress=float(envelope.max()),
        max_stress_ratio=float(envelope.max() / cs.stress_limit),
        iterations=it, converged=converged, termination=termination,
        stress_feasible=bool(max_con <= op.feasibility_tol) if ev.uses_stress else True,
        max_constraint=max_con, history=hist, wall_seconds=time.time() - t0,
    )

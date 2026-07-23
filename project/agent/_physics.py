"""Deterministic verification and quality checks for one SIMP run.

The returned top-level ``passed`` flag is controlled only by hard numerical and
physics-consistency checks. Heuristic design-quality measures (grayness,
checkerboard proxy, convergence status) are reported separately as warnings and
do not masquerade as proof that the governing equations were solved correctly.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def _require_history(metrics: dict, key: str) -> np.ndarray:
    values = metrics.get(key)
    if values is None or len(values) == 0:
        raise ValueError(f"Missing required metric history: {key}")
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite values in metric history: {key}")
    return values


def _add(checks, failures, warnings, name, passed, value, threshold, severity, message):
    checks[name] = {
        "passed": bool(passed),
        "value": value,
        "threshold": threshold,
        "severity": severity,
    }
    if not passed:
        if severity == "hard":
            failures.append(message)
        else:
            warnings.append(message)



def _add_diagnostic(checks, name, value, interpretation):
    """Record an informational metric with no pass/fail interpretation."""
    checks[name] = {
        "passed": None,
        "value": value,
        "threshold": interpretation,
        "severity": "diagnostic",
    }

def _checkerboard_index(rho_grid: np.ndarray) -> float:
    """Return a normalized alternating 2x2-mode indicator (informational)."""
    if rho_grid.ndim != 2 or min(rho_grid.shape) < 2:
        return 0.0
    mixed = np.abs(
        rho_grid[:-1, :-1]
        + rho_grid[1:, 1:]
        - rho_grid[1:, :-1]
        - rho_grid[:-1, 1:]
    )
    gx = np.abs(np.diff(rho_grid, axis=1)).mean()
    gy = np.abs(np.diff(rho_grid, axis=0)).mean()
    scale = gx + gy + 1.0e-12
    return float(np.mean(mixed) / (2.0 * scale))



def _gradient_evidence(metrics: dict, key: str, n_cells: int):
    """Return (valid, array, reason) without raising on malformed evidence."""
    raw = metrics.get(key)
    if raw is None:
        return False, None, f"{key} is missing"
    try:
        arr = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        return False, None, f"{key} cannot be converted to a numeric array: {exc}"
    if arr.shape != (n_cells,):
        return False, None, f"{key} has shape {arr.shape}; expected {(n_cells,)}"
    if not np.all(np.isfinite(arr)):
        return False, None, f"{key} contains NaN or infinity"
    return True, arr, "ok"


def _load_support_connected(rho_grid, support_mask, load_mask, threshold=0.5):
    """Four-neighbor flood fill on a thresholded physical-density grid."""
    rho_grid = np.asarray(rho_grid, dtype=float)
    support_mask = np.asarray(support_mask, dtype=bool)
    load_mask = np.asarray(load_mask, dtype=bool)
    if not (rho_grid.shape == support_mask.shape == load_mask.shape):
        raise ValueError("density/support/load masks must have identical shapes")
    solid = rho_grid >= float(threshold)
    starts = np.argwhere(solid & support_mask)
    targets = solid & load_mask
    if len(starts) == 0:
        return False, "no thresholded solid cell touches a support region"
    if not np.any(targets):
        return False, "no thresholded solid cell touches a load region"
    visited = np.zeros_like(solid, dtype=bool)
    queue = deque((int(i), int(j)) for i, j in starts)
    for i, j in queue:
        visited[i, j] = True
    nrow, ncol = solid.shape
    while queue:
        i, j = queue.popleft()
        if targets[i, j]:
            return True, "connected"
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = i + di, j + dj
            if (0 <= ni < nrow and 0 <= nj < ncol
                    and solid[ni, nj] and not visited[ni, nj]):
                visited[ni, nj] = True
                queue.append((ni, nj))
    return False, "no four-neighbor solid path joins load and support regions"


def validate(
    metrics: dict,
    rho_final: np.ndarray,
    volfrac_target: float,
    n_cells: int,
    *,
    cell_volumes: np.ndarray | None = None,
    r_min: float | None = None,
    element_size: float | None = None,
    rho_grid: np.ndarray | None = None,
    support_cell_mask: np.ndarray | None = None,
    load_cell_mask: np.ndarray | None = None,
) -> dict:
    """Run deterministic post-solve checks.

    Hard checks establish algebraic consistency and basic admissibility. Quality
    checks flag a design that may be insufficiently crisp or converged, but they
    are not treated as governing-equation validation.
    """
    checks: dict = {}
    failures: list[str] = []
    warnings: list[str] = []

    rho_final = np.asarray(rho_final, dtype=float)
    if rho_final.size != n_cells:
        failures.append(
            f"Final density has {rho_final.size} entries; expected {n_cells}."
        )
    finite_rho = bool(np.all(np.isfinite(rho_final)))
    _add(
        checks, failures, warnings,
        "density_finite", finite_rho,
        finite_rho, "all entries finite", "hard",
        "Final density contains NaN or infinity.",
    )
    if not finite_rho:
        return {
            "passed": False,
            "checks": checks,
            "failure_reasons": failures,
            "quality_warnings": warnings,
        }

    try:
        compliance = _require_history(metrics, "compliance_history")
        residuals = _require_history(metrics, "residual_history")
        ksp_reasons = _require_history(metrics, "ksp_reason_history")
        energy_errors = _require_history(metrics, "work_energy_error_history")
    except ValueError as exc:
        failures.append(str(exc))
        return {
            "passed": False,
            "checks": checks,
            "failure_reasons": failures,
            "quality_warnings": warnings,
        }

    final_compliance = float(compliance[-1])
    _add(
        checks, failures, warnings,
        "compliance_positivity", final_compliance > 0.0,
        final_compliance, "> 0", "hard",
        f"Final compliance is non-positive ({final_compliance:.6g}).",
    )

    all_ksp_ok = bool(np.all(ksp_reasons > 0))
    _add(
        checks, failures, warnings,
        "linear_solver_convergence", all_ksp_ok,
        [int(v) for v in ksp_reasons.tolist()],
        "PETSc converged reason > 0 for every solve", "hard",
        "At least one PETSc linear solve did not converge.",
    )

    final_residual = float(residuals[-1])
    _add(
        checks, failures, warnings,
        "equilibrium_residual", final_residual < 1.0e-6,
        final_residual, "< 1e-6 relative algebraic residual", "hard",
        f"Final relative residual {final_residual:.3e} exceeds 1e-6.",
    )

    final_energy_error = float(energy_errors[-1])
    _add(
        checks, failures, warnings,
        "work_energy_consistency", final_energy_error < 1.0e-8,
        final_energy_error, "|F^T U - U^T K U| / scale < 1e-8", "hard",
        f"Work-energy relative mismatch {final_energy_error:.3e} exceeds 1e-8.",
    )

    if cell_volumes is None:
        cell_volumes = np.ones_like(rho_final)
    else:
        cell_volumes = np.asarray(cell_volumes, dtype=float)
    valid_volumes = (
        cell_volumes.shape == rho_final.shape
        and np.all(np.isfinite(cell_volumes))
        and np.all(cell_volumes > 0.0)
    )
    _add(
        checks, failures, warnings,
        "cell_measure_validity", valid_volumes,
        bool(valid_volumes), "positive finite measure for every cell", "hard",
        "Cell-volume vector is invalid or does not match the density field.",
    )
    if valid_volumes:
        vf_final = float(np.dot(rho_final, cell_volumes) / cell_volumes.sum())
        vf_error = abs(vf_final - volfrac_target)
        _add(
            checks, failures, warnings,
            "volume_fraction", vf_error < 0.02,
            vf_final, f"{volfrac_target} +/- 0.02", "hard",
            f"Volume fraction {vf_final:.6g} differs from target "
            f"{volfrac_target:.6g} by {vf_error:.3e}.",
        )

    phys_valid, dc_drho_phys, phys_reason = _gradient_evidence(
        metrics, "final_dc_drho_phys", n_cells
    )
    _add(
        checks, failures, warnings,
        "dc_drho_phys_evidence", phys_valid,
        phys_reason,
        "finite array with shape (n_cells,); derivative = dC/d(rho_phys)",
        "hard",
        "Direct SIMP derivative evidence is missing, stale, malformed, or non-finite: "
        + phys_reason,
    )
    if phys_valid:
        sign_tol = max(1.0e-12, 1.0e-10 * float(np.max(np.abs(dc_drho_phys))))
        n_positive = int(np.count_nonzero(dc_drho_phys > sign_tol))
        _add(
            checks, failures, warnings,
            "dc_drho_phys_sign", n_positive == 0,
            n_positive,
            f"0 entries greater than {sign_tol:.3e}; derivative = dC/d(rho_phys)",
            "hard",
            f"{n_positive} direct physical-density compliance derivatives are positive.",
        )

    design_valid, dc_dx_design, design_reason = _gradient_evidence(
        metrics, "final_dc_dx_design", n_cells
    )
    _add(
        checks, failures, warnings,
        "dc_dx_design_evidence", design_valid,
        design_reason,
        "finite array with shape (n_cells,); derivative supplied to MMA = "
        "H^T[(dC/d(rho_phys))*(d(rho_phys)/d(rho_tilde))]",
        "hard",
        "MMA design-gradient evidence is missing, stale, malformed, or non-finite: "
        + design_reason,
    )

    bounds_ok = bool(
        rho_final.min() >= -1.0e-8 and rho_final.max() <= 1.0 + 1.0e-8
    )
    _add(
        checks, failures, warnings,
        "density_bounds", bounds_ok,
        [float(rho_final.min()), float(rho_final.max())], "[0, 1]", "hard",
        f"Density is out of bounds: [{rho_final.min():.6g}, {rho_final.max():.6g}].",
    )

    if r_min is not None and element_size is not None:
        ratio = float(r_min / element_size)
        _add(
            checks, failures, warnings,
            "filter_radius_minimum", ratio >= 1.0,
            ratio, ">= 1 element", "hard",
            f"Filter radius spans only {ratio:.3f} elements.",
        )
        _add(
            checks, failures, warnings,
            "filter_radius_oversmoothing_warning", ratio <= 6.0,
            ratio, "<= 6 elements (project heuristic)", "quality",
            f"Filter radius spans {ratio:.2f} elements and may over-smooth the topology.",
        )

    mid_gray = float(np.mean((rho_final > 0.25) & (rho_final < 0.75)))
    _add(
        checks, failures, warnings,
        "gray_fraction", mid_gray < 0.35,
        mid_gray, "< 0.35 for 0.25 < rho < 0.75 (project heuristic)", "quality",
        f"Intermediate-density fraction is {mid_gray:.1%}; binarization may be incomplete.",
    )

    if rho_grid is not None:
        grid = np.asarray(rho_grid, dtype=float)
        if grid.ndim == 2 and grid.size == n_cells and np.all(np.isfinite(grid)):
            checkerboard = _checkerboard_index(grid)
            _add_diagnostic(
                checks,
                "checkerboard_alternating_mode_index",
                checkerboard,
                (
                    "informational only; uncalibrated 2x2 alternating-mode proxy. "
                    "It can respond to legitimate diagonal/stair-stepped boundaries "
                    "and must not be described as proof of checkerboarding."
                ),
            )
            if support_cell_mask is not None and load_cell_mask is not None:
                connectivity = {}
                for threshold in (0.3, 0.5, 0.7):
                    try:
                        connected, detail = _load_support_connected(
                            grid,
                            support_cell_mask,
                            load_cell_mask,
                            threshold=threshold,
                        )
                    except (TypeError, ValueError) as exc:
                        connected, detail = False, f"invalid evidence: {exc}"
                    connectivity[f"rho_{threshold:.1f}"] = {
                        "connected": bool(connected),
                        "detail": detail,
                    }
                _add_diagnostic(
                    checks,
                    "load_to_support_connectivity_threshold_sweep",
                    connectivity,
                    (
                        "informational threshold sweep using four-neighbor cell "
                        "connectivity; no calibrated hard threshold"
                    ),
                )
        else:
            _add(
                checks, failures, warnings,
                "spatial_metric_evidence", False,
                f"shape={grid.shape}, size={grid.size}",
                "finite 2-D grid containing n_cells entries", "quality",
                "Spatial density evidence is malformed; checkerboard and connectivity "
                "diagnostics were not evaluated.",
            )

    kkt = metrics.get("final_kkt_diagnostics")
    if isinstance(kkt, dict) and kkt.get("available"):
        finite_kkt = all(
            np.isfinite(float(kkt[key])) for key in ("residual_norm", "residual_max")
        )
        if finite_kkt:
            _add_diagnostic(
                checks,
                "kkt_residual_diagnostic",
                {
                    "norm": float(kkt["residual_norm"]),
                    "max": float(kkt["residual_max"]),
                },
                (
                    "diagnostic only; last MMA-subproblem multipliers evaluated "
                    "with final re-evaluated gradients; not a convergence certificate"
                ),
            )
        else:
            warnings.append("KKT diagnostic contains non-finite values.")
    else:
        warnings.append("KKT diagnostic is unavailable.")

    design_converged = bool(metrics.get("design_converged", False))
    objective_plateau = bool(metrics.get("objective_plateau", False))
    continuation_complete = bool(metrics.get("continuation_complete", False))
    _add(
        checks, failures, warnings,
        "design_convergence_status", design_converged,
        design_converged, "max design change < configured tol_change", "quality",
        "The returned design did not satisfy the configured max-change tolerance.",
    )
    _add(
        checks, failures, warnings,
        "continuation_completion_status", continuation_complete,
        continuation_complete,
        "all effective beta steps applied and held for the required tail",
        "quality",
        "The effective continuation plan was not completed and settled.",
    )
    _add_diagnostic(
        checks,
        "objective_plateau_status",
        objective_plateau,
        (
            "informational only; objective stagnation does not establish design "
            "convergence"
        ),
    )

    return {
        "passed": len(failures) == 0,
        "checks": checks,
        "failure_reasons": failures,
        "quality_warnings": warnings,
    }

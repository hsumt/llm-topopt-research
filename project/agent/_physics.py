"""
_physics.py
"""
import numpy as np
def validate(metrics: dict,
    rho_final: np.ndarray,
    volfrac_target: float,
    n_cells: int,
    ) -> dict:
    
    # Check #1 that the compliance is monotonically decreasing after iter 5. 
    # Check that vol fraction is within the target range
    # Residual  & Sensitivity = positive
    # No checkerboards
    # Density field is all 0s and 1s
    checks = {}
    reasons = []


    # Compliance monotonic
    # history = metrics["compliance_history"]
    # if len(history) > 5:
    #     tail = history[5:]
    #     violations = sum(1 for i in range(1, len(tail)) if tail[i] > tail[i-1] * 1.02)
    #     mono_ok = violations < len(tail) * 0.1
    #     checks["compliance_monotone"] = {
    #         "passed": mono_ok,
    #         "value": violations,
    #         "threshold": f"< {len(tail) * 0.1:.1f} violations"
    #     }
    #     if not mono_ok:
    #         reasons.append(
    #             f"Compliance non-monotone: {violations} violations in {len(tail)} iters"
            
    #         )
    history = metrics["compliance_history"]
    final_compliance = float(history[-1])
    compliance_ok = final_compliance > 0.0
    checks["compliance_positivity"] = {
        "passed": compliance_ok,
        "value": round(final_compliance, 6),
        "threshold": ">0.0",
    }
    if not compliance_ok:
        reasons.append(
            f"Final compliance is non-positive ({final_compliance:.6f})"
            f"indicates a sign error in load, BCs, or assembly, not a optimization convergence issue"
        )
    # Residual
    residual_history = metrics.get("residual_history")
    if residual_history:
        final_residual = float(residual_history[-1])
        residual_ok = final_residual < 1e-5
        checks["equilibrium_residual"] = {
            "passed":    residual_ok,
            "value":     final_residual,
            "threshold": "< 1e-5 (relative)",
        }
        if not residual_ok:
            reasons.append(
                f"Relative equilibrium residual {final_residual:.3e} exceeds "
                f"tolerance — final solve may not satisfy Ku=F to the "
                f"precision the optimizer assumes"
            )
    else:
        checks["equilibrium_residual"] = {
            "passed":    False,
            "value":     None,
            "threshold": "< 1e-5 (relative)",
        }
        reasons.append(
            "residual_history missing from metrics — equilibrium residual "
            "was not checked. This is a plumbing gap, not a physics failure; "
            "fix the caller before treating this as a real validation failure."
        )
    #Vol Fraction check
    vf_final = float(rho_final.sum() / n_cells) # checks the volume fraction of the final mesh
    vf_err = abs(vf_final - volfrac_target) #grabs the error value as an abs
    vf_ok = vf_err < 0.02 #checks within 0.02 of the volume fraction
    checks["volume_fraction"] = {
        "passed": vf_ok,
        "value": round(vf_final, 4),
        "threshold": f"{volfrac_target} ± 0.02"
    }
    if not vf_ok:
        reasons.append(
            f"Volume fraction {vf_final:.4f} outside tolerance"
            f"target {volfrac_target} ± 0.02"
        )
    
    # Sensitivity sign

    final_sensitivity = metrics.get("final_sensitivity")
    if final_sensitivity is not None:
        final_sensitivity = np.asarray(final_sensitivity)
        n_violations = int(np.sum(final_sensitivity > 1e-8))
        sens_ok = n_violations == 0
        checks["sensitivity_sign"] = {
            "passed":    sens_ok,
            "value":     n_violations,
            "threshold": "0 positive entries (dc/drho <= 0 everywhere)",
        }
        if not sens_ok:
            reasons.append(
                f"{n_violations} elements have dc/drho > 0 violates "
                f"Sigmund (2001) Eq. 4 sign requirement; check strain-energy "
                f"sign convention in compute_sensitivities"
            )
    else:
        checks["sensitivity_sign"] = {
            "passed":    False,
            "value":     None,
            "threshold": "0 positive entries (dc/drho <= 0 everywhere)",
        }
        reasons.append(
            "final_sensitivity missing from metrics. Sensitivity sign was "
            "not checked. Robustness gap, not a physics failure."
        )
    #Check [0,1] density field. Checks if density field exceeds 1.000001
    bounds_ok = bool(rho_final.min() >= -1e-3 and rho_final.max() <= 1.0 + 1e-3)
    checks["density_bounds"] = {
        "passed":    bounds_ok,
        "value":     f"[{rho_final.min():.4f}, {rho_final.max():.4f}]",
        "threshold": "[0.0, 1.0]"
    }
    if not bounds_ok:
        reasons.append(
            f"Density out of bounds: min={rho_final.min():.4f}, "
            f"max={rho_final.max():.4f}"
        )
    
    #check for checkerboarding
    # Proxy: fraction of elements that are "grey" (0.1 < rho < 0.9)
    # A well-converged SIMP result should be mostly 0/1 with some grey
    # at boundaries. If >60% grey the filter is definitely not working and we can immediately send it back.
    # grey_fraction = float(np.mean((rho_final > 0.1) & (rho_final < 0.9)))
    # grey_ok = grey_fraction < 0.6
    # checks["grey_fraction"] = {
    #     "passed":    grey_ok,
    #     "value":     round(grey_fraction, 4),
    #     "threshold": "< 0.60"
    # }
    # if not grey_ok:
    #     reasons.append(
    #         f"Excessive grey elements ({grey_fraction:.1%}) — "
    #         f"filter may be ineffective or penal too low"
    #     )


    mid_grey_fraction = float(np.mean((rho_final > 0.25) & (rho_final < 0.75)))
    grey_ok = mid_grey_fraction < 0.35
    checks["grey_fraction"] = {
        "passed":    grey_ok,
        "value":     round(mid_grey_fraction, 4),
        "threshold": "< 0.35 for 0.25 < rho < 0.75"
    }
    if not grey_ok:
        reasons.append(
            f"Excessive grey elements ({mid_grey_fraction:.1%}) — "
            f"filter may be ineffective or penal too low"
        )
    # # Check for structural connectivity [to do]
    # conn = _check_connectivity(rho_final, nelx, nely)
    # checks["structural_connectivity"] = conn
    # if not conn["passed"]:
    #     reasons.append(
    #         f"Structural connectivity failed: {conn['value']}"
    #     )

    # return {
    #     "passed":          len(reasons) == 0,
    #     "checks":          checks,
    #     "failure_reasons": reasons,
    # }



    return {
        "passed":          len(reasons) == 0,
        "checks":          checks,
        "failure_reasons": reasons,
    }
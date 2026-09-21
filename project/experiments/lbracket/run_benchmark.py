"""Holmberg L-bracket benchmark: formulations P1, P2 and P3.

Reproduces the comparison of Holmberg et al. (2013) Sec. 9.3 / Table 5 at a
chosen fidelity tier, and additionally runs a *bounded* probe at the paper's own
discretisation (h = 2 mm, 6400 elements) so the low-fidelity tier can be quoted
against the reference cost without paying for a full reference optimization.

    python -m project.experiments.lbracket.run_benchmark --fidelity low

The mass limit used by P2 and P3 is taken from the converged P1 mass when P1 is
run, which is how the paper sets it up ("the mass that was found to be optimal
is used as limit value for the mass constraint in formulations (P2) and (P3)").
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from project.topopt.stress.problem import FIDELITY, elements_at_fidelity, spec_at_fidelity
from project.topopt.stress.solver import Evaluator, solve_stress_problem

DEFAULT_OUT = Path("artifacts/benchmark")


def reference_probe(formulations=("P3", "P1"), max_iter: int = 1) -> list:
    """Cost probe at Holmberg's own discretisation, capped at ``max_iter``.

    This exists to state what the reference tier costs per iteration on the
    machine actually running it, not to produce a converged reference design.
    """
    rows = []
    for form in formulations:
        spec = spec_at_fidelity("reference", max_iter=max_iter)
        spec.formulation = form
        t = time.time()
        ev = Evaluator(spec)
        ev.evaluate(ev.initial_design())
        rows.append({
            "tier": "reference",
            "formulation": form,
            "n_elements": int(ev.ne),
            "n_dof": int(ev.mesh.n_dof),
            "element_size_mm": ev.mesh.geom.h,
            "seconds_per_iteration": time.time() - t,
        })
    return rows


#: All three formulations get the same iteration budget so the comparison is not
#: confounded by termination point.  The stress-constrained ones need it:
#: reclustering every iteration keeps the constraint set moving, so the design
#: change falls below tolerance far later than for the stiffness-only problem.
BENCHMARK_ITERATION_BUDGET = 400


def run(fidelity: str = "low", formulations=("P1", "P2", "P3"),
        out_dir: Path | None = None, probe: bool = True) -> dict:
    out_dir = Path(out_dir or DEFAULT_OUT)
    out_dir.mkdir(parents=True, exist_ok=True)

    results, rows = {}, []
    mass_fraction_from_P1 = None
    p1_termination = None

    for form in formulations:
        spec = spec_at_fidelity(fidelity)
        spec.formulation = form
        if form in ("P2", "P3") and mass_fraction_from_P1 is not None:
            spec.constraints.mass_fraction_limit = mass_fraction_from_P1
        spec.optimizer.max_iter = max(spec.optimizer.max_iter, BENCHMARK_ITERATION_BUDGET)
        res = solve_stress_problem(spec)
        results[form] = res
        if form == "P1":
            mass_fraction_from_P1 = float(res.mass_fraction)
            p1_termination = (res.termination, bool(res.converged), bool(res.stress_feasible))
        s = res.summary()
        if form in ("P2", "P3") and mass_fraction_from_P1 is not None:
            term, conv, feas = p1_termination
            s["mass_limit_source"] = (
                "P1 final mass; P1 converged and was feasible" if conv else
                f"P1 final mass at termination ({term}); P1 did NOT meet the design-change "
                f"tolerance, stress_feasible={feas}. The limit therefore traces to P1's last "
                f"design, not to a converged optimum."
            )
        else:
            s["mass_limit_source"] = "specification default"
        rows.append(s)
        print(json.dumps(s, indent=2, default=str))

    payload = {
        "fidelity": fidelity,
        "n_elements": elements_at_fidelity(fidelity),
        "grid": FIDELITY[fidelity],
        "formulations": rows,
        "reference_probe": reference_probe() if probe else [],
    }
    path = out_dir / f"benchmark_{fidelity}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {path}")
    return {"payload": payload, "results": results}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fidelity", default="low", choices=sorted(FIDELITY))
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-probe", action="store_true")
    a = ap.parse_args()
    run(fidelity=a.fidelity, out_dir=a.out, probe=not a.no_probe)
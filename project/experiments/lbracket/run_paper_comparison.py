"""Quantitative comparison against Holmberg et al. (2013) at the paper's own mesh.

The low-fidelity tier is where the cascade runs; it is NOT a basis for claiming
agreement with the paper.  This module reproduces the paper's own configurations
at its own discretisation (6400 elements, h = 2 mm) so the published numbers can
be compared directly:

  Table 1  L-beam, P1, "stress level" clustering, reclustered every iteration
           M = 24.76e-3 kg,  C = 10,330 N mm
  Table 2  L-beam, P1, "distributed stress" clustering, every iteration
           M = 20.63e-3 kg,  C = 13,277 N mm
  Table 5  L-beam, comparison of formulations at the Table 2 mass
           C(P1) = 13,276,  C(P2) = 14,347,  C(P3) = 10,960 N mm

Run:  python -m project.experiments.lbracket.run_paper_comparison
"""
from __future__ import annotations

import json
from pathlib import Path

from project.topopt.stress.problem import spec_at_fidelity
from project.topopt.stress.solver import solve_stress_problem

FULL_MASS_G = 71.68     # 25600 mm^2 x 1 mm x 2.8e-9 ton/mm^3, the paper's L-beam

PUBLISHED = {
    "table1_P1_stress_level":       {"mass_g": 24.76, "compliance": 10330},
    "table2_P1_distributed_stress": {"mass_g": 20.63, "compliance": 13277},
    "table5_P3":                    {"mass_g": 20.63, "compliance": 10960},
    "table5_P2":                    {"mass_g": 20.63, "compliance": 14347},
}


def _row(label, res, published):
    m_g = res.mass * 1e6            # ton -> g
    out = {
        "case": label,
        "n_elements": int(res.mesh.n_elem),
        "iterations": res.iterations,
        "converged": bool(res.converged),
        "termination": res.termination,
        "mass_g": m_g,
        "mass_fraction": float(res.mass_fraction),
        "compliance_N_mm": float(res.compliance),
        "max_von_mises_MPa": float(res.max_stress),
        "stress_feasible": bool(res.stress_feasible),
        "published_mass_g": published["mass_g"],
        "published_compliance_N_mm": published["compliance"],
        "mass_ratio_vs_paper": m_g / published["mass_g"],
        "compliance_ratio_vs_paper": float(res.compliance) / published["compliance"],
    }
    return out


def run(tier: str = "reference", max_iter: int = 400, out_dir: str | None = None) -> dict:
    rows = []

    s = spec_at_fidelity(tier, max_iter=max_iter)
    s.formulation = "P1"
    s.constraints.clustering = "stress_level"
    r1 = solve_stress_problem(s, record_history=False)
    rows.append(_row("table1_P1_stress_level", r1, PUBLISHED["table1_P1_stress_level"]))

    s = spec_at_fidelity(tier, max_iter=max_iter)
    s.formulation = "P1"
    s.constraints.clustering = "distributed_stress"
    r2 = solve_stress_problem(s, record_history=False)
    rows.append(_row("table2_P1_distributed_stress", r2,
                     PUBLISHED["table2_P1_distributed_stress"]))

    # Table 5 sets the mass limit for P2 and P3 from the Table 2 P1 optimum.
    mf = PUBLISHED["table2_P1_distributed_stress"]["mass_g"] / FULL_MASS_G
    for form in ("P3", "P2"):
        s = spec_at_fidelity(tier, max_iter=max_iter)
        s.formulation = form
        s.constraints.mass_fraction_limit = mf
        s.constraints.clustering = "distributed_stress"
        r = solve_stress_problem(s, record_history=False)
        rows.append(_row(f"table5_{form}", r, PUBLISHED[f"table5_{form}"]))

    payload = {"tier": tier, "max_iter": max_iter,
               "paper_mass_fraction_for_table5": mf, "rows": rows}
    d = Path(out_dir or "artifacts/benchmark")
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"paper_comparison_{tier}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="reference",
                    help="fidelity tier to compare at; the published numbers are at 'reference'")
    ap.add_argument("--max-iter", type=int, default=400)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(tier=a.tier, max_iter=a.max_iter, out_dir=a.out)

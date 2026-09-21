"""Predicate-set ablation for the ATO cascade on the L-bracket.

The anomaly predicate is the load-bearing design decision: the cascade's
partition is exact only if the predicate is.  This experiment holds everything
else fixed -- the same stated problem, the same generated candidate set, the
same re-solves -- and varies only which predicates the cascade is asked to
discharge, so the outcome difference is attributable to the predicate alone.

Candidates are generated once and injected into every configuration.  Re-solves
are cached across configurations, because a subset's design depends on the edits
and not on which predicates are being targeted.

    python -m project.experiments.lbracket.run_predicate_ablation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from project.ato.anomaly import ALL_PREDICATES, PEAK_PREDICATE, detect
from project.ato.cascade import run_cascade
from project.ato.hypothesize import propose
from project.experiments.lbracket.run_ato import build_spec
from project.topopt.stress.solver import solve_stress_problem

CONFIGS = {
    "A_peak_only": (PEAK_PREDICATE,),
    "B_extent_and_features": ALL_PREDICATES,
    "C_extent_only": ("widespread_overstress",),
}


def main(backend=None, *, fidelity: str = "low", max_candidates: int = 5,
         max_subset_size: int = 3, discharge_max_iter: int = 200,
         out_dir: str | None = None) -> dict:
    if backend is None:
        from dotenv import load_dotenv
        from project.ato.backends import AnthropicBackend
        load_dotenv()
        backend = AnthropicBackend()

    spec = build_spec(fidelity)
    baseline = solve_stress_problem(spec)
    if not baseline.converged:
        raise RuntimeError(f"baseline did not converge: {baseline.termination}")

    full = detect(baseline, active=tuple([PEAK_PREDICATE]) + ALL_PREDICATES)
    gen = propose(backend, spec, full, baseline.summary(), max_candidates=max_candidates)
    candidates = gen["candidates"]
    print(f"generated {len(candidates)} candidates, reused across every configuration")

    cache, out = {}, {}
    for name, active in CONFIGS.items():
        print(f"\n=== configuration {name}: targets {list(active)} ===")
        out[name] = run_cascade(
            spec, backend, candidates=candidates, active_predicates=active,
            max_subset_size=max_subset_size, discharge_max_iter=discharge_max_iter,
            solve_cache=cache,
        )

    payload = {
        "fidelity": fidelity,
        "baseline": baseline.summary(),
        "full_evidence": full.to_dict(),
        "generation": {"model": gen["model"], "usage": gen["usage"],
                       "candidates": candidates, "rejected": gen["rejected"]},
        "configurations": {k: {"targets": list(CONFIGS[k]),
                               "outcome": v["outcome"],
                               "statement": v["statement"],
                               "counts": v["ranking"]["counts"],
                               "ranking": v["ranking"],
                               "discharges": v["discharges"]}
                           for k, v in out.items()},
    }
    p = Path(out_dir or "artifacts/ato")
    p.mkdir(parents=True, exist_ok=True)
    path = p / f"predicate_ablation_{fidelity}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {path}")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fidelity", default="low")
    ap.add_argument("--max-candidates", type=int, default=5)
    ap.add_argument("--max-subset-size", type=int, default=3)
    ap.add_argument("--discharge-max-iter", type=int, default=200)
    a = ap.parse_args()
    main(fidelity=a.fidelity, max_candidates=a.max_candidates,
         max_subset_size=a.max_subset_size, discharge_max_iter=a.discharge_max_iter)

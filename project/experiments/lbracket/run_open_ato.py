"""Run the open ATO cascade on the L-bracket.

    python -m project.experiments.lbracket.run_open_ato --max-rounds 2

Blind hypothesis generation, capability matching, mid-run tool synthesis,
deductive discharge, authority ruling against the declared intent, and closure.
The default backend reads ANTHROPIC_API_KEY from the process environment; this
module never reads the .env file itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from project.ato.intent import DesignIntent
from project.ato.open_cascade import run_iterative_cascade
from project.topopt.stress.problem import spec_at_fidelity

DEFAULT_OUT = Path("artifacts/ato")
INTENT_PATH = Path("project/experiments/lbracket/intent_lbracket.json")


def build_spec(fidelity: str = "low", formulation: str = "P3"):
    spec = spec_at_fidelity(fidelity)
    spec.formulation = formulation
    return spec


def main(backend=None, *, fidelity: str = "low", formulation: str = "P3",
         max_subset_size: int = 3, discharge_max_iter: int = 120,
         max_rounds: int = 2, max_hypotheses: int = 6,
         accept_max_iter: int = 200, max_epochs: int = 3,
         discharge_n_cells: int | None = None,
         intent_path=None, out_dir=None) -> dict:
    if backend is None:
        from project.ato.backends import AnthropicBackend
        backend = AnthropicBackend()
    intent = DesignIntent.load(intent_path or INTENT_PATH)
    spec = build_spec(fidelity, formulation)
    report = run_iterative_cascade(
        spec, backend, intent,
        max_epochs=max_epochs, max_subset_size=max_subset_size,
        discharge_max_iter=discharge_max_iter, max_rounds=max_rounds,
        max_hypotheses=max_hypotheses, accept_max_iter=accept_max_iter,
        discharge_n_cells=discharge_n_cells,
    )
    out = Path(out_dir or DEFAULT_OUT)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"open_ato_{fidelity}_{formulation}.json"
    path.write_text(json.dumps({k: v for k, v in report.items() if k != "_results"},
                               indent=2, default=str))
    print(f"\\nwrote {path}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fidelity", default="low")
    ap.add_argument("--formulation", default="P3")
    ap.add_argument("--max-subset-size", type=int, default=3)
    ap.add_argument("--discharge-max-iter", type=int, default=120)
    ap.add_argument("--max-rounds", type=int, default=2)
    ap.add_argument("--max-hypotheses", type=int, default=6)
    ap.add_argument("--accept-max-iter", type=int, default=200)
    ap.add_argument("--max-epochs", type=int, default=3)
    ap.add_argument("--discharge-n-cells", type=int, default=None,
                    help="coarsen the in-plane mesh for discharge re-solves only")
    ap.add_argument("--intent", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(fidelity=a.fidelity, formulation=a.formulation,
         max_subset_size=a.max_subset_size, discharge_max_iter=a.discharge_max_iter,
         max_rounds=a.max_rounds, max_hypotheses=a.max_hypotheses,
         accept_max_iter=a.accept_max_iter, max_epochs=a.max_epochs,
         discharge_n_cells=a.discharge_n_cells,
         intent_path=a.intent, out_dir=a.out)

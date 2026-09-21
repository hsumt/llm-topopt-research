"""Run the ATO cascade on the Holmberg L-bracket.

Default backend is the Anthropic Messages API, which reads ANTHROPIC_API_KEY
from the process environment.  The key is loaded with ``dotenv.load_dotenv()``;
this module never reads the .env file itself.

    python -m project.experiments.lbracket.run_ato --fidelity low

Pass ``backend=`` to ``main`` to drive the same cascade from a host that
supplies its own model access.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from project.ato.cascade import run_cascade
from project.topopt.stress.problem import spec_at_fidelity

DEFAULT_OUT = Path("artifacts/ato")

PING_TOOL = {
    "name": "ping",
    "description": "Acknowledge by returning ok=true.",
    "input_schema": {"type": "object",
                     "properties": {"ok": {"type": "boolean"}},
                     "required": ["ok"]},
}


def check_backend() -> dict:
    """Validate the Anthropic key path with one minimal call.

    The cascade's results in ``RESULTS_L_bracket_ATO.md`` were produced through an
    injected backend, so this code path had not been exercised against a live key.
    Run this once before a real cascade: it costs a few hundred tokens and confirms
    the key resolves, the tool-use round trip works, and usage accounting reports.
    """
    from dotenv import load_dotenv

    from project.ato.backends import AnthropicBackend

    load_dotenv()
    backend = AnthropicBackend()
    out = backend("Acknowledge by calling the tool.", "Call ping with ok=true.", PING_TOOL)
    print(f"backend OK  model={backend.model}  usage={backend.usage}  returned={out}")
    return {"model": backend.model, "usage": backend.usage, "returned": out}


def build_spec(fidelity: str = "low", formulation: str = "P3"):
    """The stated problem the cascade is asked to diagnose.

    P3 -- minimum compliance subject to a mass limit -- is the traditional
    stiffness-based statement.  Holmberg Sec. 9.3 reports that it places
    material in the re-entrant corner and produces a geometric stress
    singularity, while remaining a correct answer to the problem as posed.
    That is precisely a converged, feasible run whose *specification* is at
    fault, so it is the natural baseline for adequacy diagnosis.
    """
    spec = spec_at_fidelity(fidelity)
    spec.formulation = formulation
    return spec


def main(backend=None, *, fidelity: str = "low", formulation: str = "P3",
         max_candidates: int = 6, max_subset_size: int = 2,
         discharge_max_iter: int = 150, max_rounds: int = 1,
         out_dir: Path | None = None) -> dict:
    if backend is None:
        from dotenv import load_dotenv
        from project.ato.backends import AnthropicBackend
        load_dotenv()
        backend = AnthropicBackend()

    spec = build_spec(fidelity, formulation)
    report = run_cascade(
        spec, backend,
        max_candidates=max_candidates,
        max_subset_size=max_subset_size,
        discharge_max_iter=discharge_max_iter,
        max_rounds=max_rounds,
    )
    out_dir = Path(out_dir or DEFAULT_OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_multiround" if max_rounds > 1 else ""
    path = out_dir / f"ato_cascade_{fidelity}_{formulation}{suffix}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {path}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fidelity", default="low")
    ap.add_argument("--formulation", default="P3")
    ap.add_argument("--max-candidates", type=int, default=6)
    ap.add_argument("--max-subset-size", type=int, default=2)
    ap.add_argument("--discharge-max-iter", type=int, default=150)
    ap.add_argument("--max-rounds", type=int, default=1,
                    help="cascade depth; >1 re-invokes the generator with the previous "
                         "round's verdicts (the reported result used 3)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--check-backend", action="store_true",
                    help="make one minimal API call to validate the key path, then exit")
    a = ap.parse_args()
    if a.check_backend:
        check_backend()
        raise SystemExit(0)
    main(fidelity=a.fidelity, formulation=a.formulation,
         max_candidates=a.max_candidates, max_subset_size=a.max_subset_size,
         discharge_max_iter=a.discharge_max_iter, max_rounds=a.max_rounds,
         out_dir=a.out)

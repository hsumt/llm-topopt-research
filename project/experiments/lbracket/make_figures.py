"""Write the deliverable figures and records for an iterative cascade run.

Called in-process straight after a run, because the design-state figure needs the
``RunResult`` objects (fields, not summaries) that the JSON record cannot carry:

    from project.experiments.lbracket.run_open_ato import main as run_open
    from project.experiments.lbracket.make_figures import write_all
    report = run_open(backend=..., fidelity="low")
    write_all(report)

Everything lands in ``results/L_bracket_ATO/`` -- a tracked folder, unlike
``artifacts/``, because these are the deliverables of the experiment rather than
scratch output.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from project.ato.plots import hypothesis_graph, solutions_figure

RESULTS_DIR = Path("results/L_bracket_ATO")


def _style():
    """Use the house figure style if it is importable; plain matplotlib otherwise."""
    try:
        from figure_style import apply_figure_style, panel_letter  # type: ignore
        return apply_figure_style, panel_letter
    except Exception:
        return None, None


def write_all(report: dict, out_dir=None, *, style=None, panel_letter=None) -> dict:
    out = Path(out_dir or RESULTS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    if style is None:
        style, panel_letter = _style()

    results = report.get("_results") or []
    states = report["design_states"]
    written = {}

    if results:
        p = out / "ato_solutions.png"
        solutions_figure(states, results, path=p, apply_style=style,
                         panel_letter=panel_letter)
        written["solutions"] = str(p)

        # Design fields, so a figure can be redrawn without re-solving.
        fields = {}
        for i, r in enumerate(results, start=1):
            fields[f"epoch{i}_rho"] = r.rho_elem
            fields[f"epoch{i}_svm"] = r.sigma_vm
            fields[f"epoch{i}_centroids"] = r.centroids2d
            for name, arr in r.sigma_vm_by_case.items():
                fields[f"epoch{i}_svm_{name}"] = arr
        np.savez_compressed(out / "design_fields.npz", **fields)
        written["fields"] = str(out / "design_fields.npz")

    # Tier lives on the OPERATION and the synthesized registry is on the top-level
    # report, not on each epoch record. Passing it explicitly is necessary: without
    # it the graph silently drops the tier mark for every operation the run wrote
    # itself, which is exactly the set whose admissibility the figure is about.
    tier_map = {n: v.get("tier") for n, v in
                (report.get("synthesized_operations") or {}).items()}
    for rec in report["epochs"]:
        if not rec.get("discharges"):
            continue
        ep = rec.get("epoch", 1)
        applied = None
        for a in report["applied_edit_sets"]:
            if a["epoch"] == ep:
                applied = a["ids"]
        p = out / f"ato_hypothesis_graph_epoch{ep}.png"
        hypothesis_graph(rec, path=p, applied_ids=applied, apply_style=style,
                         tier_map=tier_map)
        written[f"graph_epoch{ep}"] = str(p)

    rec_path = out / "open_ato_run.json"
    rec_path.write_text(json.dumps(
        {k: v for k, v in report.items() if k != "_results"}, indent=2, default=str))
    written["record"] = str(rec_path)

    src = out / "synthesized_operations.py"
    blocks = ["# Operations written by the synthesis stage during this run.\n"
              "# Recorded verbatim for review. NOT registered anywhere: a synthesized\n"
              "# operation lives only for the run that wrote it.\n"]
    for rec in report["epochs"]:
        for s in rec.get("synthesis", []):
            w = s.get("written") or {}
            head = (f"\n# --- {s.get('hypothesis_id')} -> {w.get('operation_name', 'REJECTED')}"
                    f"  admitted={s.get('admitted')}")
            if s.get("admitted"):
                head += (f"  slot={w.get('slot')!r}  tier={s.get('tier')}"
                         f"  (writer proposed {s.get('proposed_tier')})")
            else:
                head += f"  rejected_because={s.get('rejected_because')!r}"
            blocks.append(head + " ---\n" + (w.get("apply_source") or "# no source returned\n"))
    src.write_text("\n".join(blocks))
    written["synthesized_source"] = str(src)
    return written

"""Orchestration of the five-stage ATO cascade."""
from __future__ import annotations

import json
import time

from project.ato.anomaly import detect
from project.ato.discharge import run_discharge
from project.ato.hypothesize import propose, propose_next_round
from project.ato.rank import rank
from project.topopt.stress.solver import solve_stress_problem


class ConvergenceGateRefusal(RuntimeError):
    """The baseline run is not eligible for specification-adequacy diagnosis."""


def run_cascade(base_spec, backend, *, max_candidates: int = 6, max_subset_size: int = 2,
                discharge_max_iter: int | None = None, active_predicates: tuple | None = None,
                candidates: list | None = None, solve_cache: dict | None = None,
                max_rounds: int = 1, verbose: bool = True) -> dict:
    t0 = time.time()
    log = print if verbose else (lambda *a, **k: None)

    log("[1/5] convergence gate: solving the stated problem")
    baseline = solve_stress_problem(base_spec)
    summary = baseline.summary()
    log(f"      {summary['termination']}  iterations={summary['iterations']}  "
        f"compliance={summary['compliance_N_mm']:.1f} N mm  "
        f"mass fraction={summary['mass_fraction']:.3f}")

    if not baseline.converged:
        raise ConvergenceGateRefusal(
            f"the stated problem terminated as {baseline.termination!r}. A run that did not "
            "converge on a feasible stated problem belongs to specification *consistency* "
            "repair, not adequacy diagnosis; the cascade refuses it."
        )

    log("[2/5] anomaly detection")
    report = detect(baseline, active=active_predicates)
    for a in report.anomalies:
        mark = "FIRED " if a.fired else ("      " if a.active else "[off] ")
        log(f"      {mark} {a.id}: {a.statement}")
    if not report.any_fired:
        return {
            "outcome": "no_anomaly",
            "statement": "The converged design triggers no anomaly predicate; "
                         "there is nothing to diagnose.",
            "baseline": summary,
            "anomaly": report.to_dict(),
            "wall_seconds": time.time() - t0,
        }

    targets = report.fired_ids()
    if candidates is not None:
        log("[3/5] hypothesis generation: reusing an injected candidate set")
        gen = {"model": "injected", "usage": {}, "candidates": list(candidates), "rejected": []}
    else:
        log("[3/5] hypothesis generation (model proposes; it does not decide verdicts)")
        gen = propose(backend, base_spec, report, summary, max_candidates=max_candidates)
    cands = gen["candidates"]
    if not cands:
        raise RuntimeError("hypothesis generation returned no grammar-valid candidate")
    for c in cands:
        log(f"      {c['id']}: {c['operation']}({json.dumps(c['parameters'], sort_keys=True)}) "
            f"[{c['tier']}, {c['boundedness']}]")
    if gen["rejected"]:
        log(f"      {len(gen['rejected'])} candidate(s) rejected by grammar validation")

    def progress(i, n, d):
        extra = f"  cleared={d.cleared}" if d.cleared and d.verdict != "SURVIVES" else ""
        log(f"      {i:>3}/{n}  {'+'.join(d.ids):<12} {d.verdict}{extra}")

    rounds = []
    all_cands = list(cands)
    discharges = []
    for rnd in range(1, max_rounds + 1):
        log(f"[4/5] deductive discharge, round {rnd}: singletons and subsets "
            f"up to size {max_subset_size}")
        discharges = run_discharge(
            base_spec, all_cands, targets,
            max_subset_size=max_subset_size, max_iter=discharge_max_iter,
            active=active_predicates, cache=solve_cache, progress=progress,
        )
        ranking = rank(discharges, all_cands)
        rounds.append({"round": rnd,
                       "candidates": [c["id"] for c in all_cands],
                       "outcome": ranking["outcome"],
                       "counts": ranking["counts"]})
        log(f"      round {rnd} outcome: {ranking['outcome']}")

        resolved = ranking["outcome"] in ("diagnosis", "flagged_intent_altering")
        if resolved or rnd == max_rounds:
            break

        log(f"[3/5] hypothesis generation, round {rnd + 1}: "
            f"informed by the round-{rnd} verdicts")
        nxt = propose_next_round(backend, base_spec, report, summary,
                                 [d.to_dict() for d in discharges],
                                 max_candidates=max_candidates)
        # Carry forward any round-n candidate that cleared part of the anomaly set:
        # a partial clearer is a component of a coupled explanation.
        partial = {i for d in discharges if d.cleared for i in d.ids}
        keep = [c for c in all_cands if c["id"] in partial][: max(0, max_candidates - 1)]
        fresh = []
        used = {c["id"] for c in keep}
        for j, c in enumerate(nxt["candidates"], 1):
            cid = f"R{rnd + 1}C{j}"
            while cid in used:
                cid += "b"
            c = dict(c, id=cid)
            used.add(cid)
            fresh.append(c)
        if not fresh:
            log("      no new grammar-valid candidate; stopping")
            break
        all_cands = keep + fresh
        gen["candidates"] = all_cands
        gen.setdefault("rounds", []).append(
            {"round": rnd + 1, "model": nxt["model"], "rejected": nxt["rejected"]})
        for c in fresh:
            log(f"      {c['id']}: {c['operation']}"
                f"({json.dumps(c['parameters'], sort_keys=True)}) "
                f"[{c['tier']}, {c['boundedness']}]")
        if keep:
            log(f"      carried forward (partial clearers): {[c['id'] for c in keep]}")
        cands = all_cands

    log("[5/5] ranking and authority gate")
    ranking = rank(discharges, all_cands)
    log(f"      {ranking['outcome']}: {ranking['statement']}")

    return {
        "outcome": ranking["outcome"],
        "statement": ranking["statement"],
        "specification": base_spec.to_dict(),
        "baseline": summary,
        "anomaly": report.to_dict(),
        "targets": targets,
        "generation": {
            "model": gen["model"],
            "usage": gen["usage"],
            "candidates": all_cands,
            "rejected": gen["rejected"],
        },
        "rounds": rounds,
        "discharges": [d.to_dict() for d in discharges],
        "ranking": ranking,
        "wall_seconds": time.time() - t0,
    }

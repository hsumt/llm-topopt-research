"""The open cascade: hypotheses first, tools second, missing tools written mid-run.

Difference from ``cascade.py``
-----------------------------
``cascade.py`` hands the generator the edit grammar and asks it to select. That caps
the hypothesis space at whatever the toolbox contains, and the symptom is that every
proposal looks like parameter tuning.

Here the two questions are separated, and the loop does not stop at a diagnosis: it
applies what the declared intent authorises and re-diagnoses, until either every
predicate is quiet or nothing further is authorised.

  per epoch
    1. convergence gate       the current stated problem must converge feasibly
    2. anomaly detection      deterministic predicates, plus a specification
                              completeness check against declared intent
    3a. blind hypothesis      what would have to be different about the stated
                              problem -- generated WITHOUT the operation catalogue
    3b. capability match      does a tool for that already exist?
    3c. tool synthesis        where it does not, a model writes one, admitted only
                              after a static gate and a smoke test
    4. deductive discharge    apply each candidate and each subset, re-solve
    5. ranking                lexicographic by tier, then edit count
    6. authority              rule on intent-altering survivors against the written
                              intent document, never on engineering plausibility
    7. accept                 apply the minimal authorised edit set; its result
                              becomes the next epoch's stated problem

Invariants that hold regardless of what any model returns:

* admissibility tier is assigned from the problem-tuple slot and the declared
  intent, never by the agent proposing the edit;
* a verdict comes only from re-solving with the verified solver;
* an intent-altering edit is applied only where the written intent authorises that
  class of change, and otherwise escalates.
"""
from __future__ import annotations

import inspect
import json
import time

from project.ato.anomaly import ALL_PREDICATES, detect
from project.ato.authority import rule as authority_rule
from project.ato.blind_hypothesize import propose_blind, untried_permitted_quantities
from project.ato.capability import match as capability_match
from project.ato.coverage import uncovered_targets
from project.ato.discharge import run_discharge
from project.ato.grammar import SYNTHESIZED, all_operations, apply_edits
from project.ato.rank import rank
from project.ato.synthesize import synthesize
from project.topopt.stress.solver import solve_stress_problem


class ConvergenceGateRefusal(RuntimeError):
    """The run is not eligible for specification-adequacy diagnosis."""


def _discharge(base_spec, candidates, targets, **kw):
    ok = set(inspect.signature(run_discharge).parameters)
    return run_discharge(base_spec, candidates, targets,
                         **{k: v for k, v in kw.items() if k in ok})


def _design_state(result) -> dict:
    """Everything needed to plot and audit one accepted design state."""
    return {
        "summary": result.summary(),
        "peak_by_case_MPa": {k: float(v.max()) for k, v in result.sigma_vm_by_case.items()},
        "stress_limit_MPa": float(result.spec.constraints.stress_limit),
        "within_allowable_by_case": {
            k: bool(v.max() <= result.spec.constraints.stress_limit)
            for k, v in result.sigma_vm_by_case.items()},
        "model": result.model,
        "n_layers": int(result.n_layers),
        "load_cases": [{"name": lc.name, "direction": lc.direction,
                        "magnitude": lc.magnitude} for lc in result.spec.load_cases],
    }


def pick_edit_set(ranking, rulings, candidates, *, discharges=None, targets=None,
                  registry=None) -> dict | None:
    """Choose the minimal edit set this system is entitled to apply.

    Three acceptance routes, in order, and the reason for each:

    1. **Full survivor.** An intent-preserving or intent-revealing (T1/T2) edit set
       that clears every fired anomaly. Applicable on the evidence alone, because it
       either changes nothing the engineer asked for or records something they
       already hold.
    2. **Authority-adopted.** An intent-altering (T3) survivor where the written
       intent authorises that class of change. The authority is the engineer's,
       exercised through their document.
    3. **Disclosure step.** A T1/T2 set that clears at least one anomaly but leaves
       or reveals others.

    Route 3 needs justifying, because it accepts an edit that makes the anomaly
    report *worse*. Demanding that one edit set clear every fired anomaly at once is
    the wrong bar: the anomalies here belong to different defects, and requiring a
    single edit to fix all of them reports "no survivor" while discarding edits that
    demonstrably fixed one of them.

    Nor is "introduces no new anomaly" the right guard. Adding an omitted load case
    reveals overstress that was always there and simply could not be seen -- the
    design was already unsafe and the specification was hiding it. An edit that
    *represents more* (T1) or *states a requirement the engineer holds* (T2) discloses
    pre-existing inadequacy rather than causing it, which is exactly what
    intent-revealing means. So new anomalies are permitted on this route and recorded
    as disclosed, while the route stays closed to T3: an intent-altering edit that
    loosens a requirement and creates new overstress is a genuine regression, and
    only the authority stage may admit one.
    """
    if not ranking:
        return None
    applicable = ranking.get("applicable") or []
    if applicable:
        best = min(applicable, key=lambda r: (len(r["ids"]), r["ids"]))
        return {"ids": best["ids"], "tier": best.get("tier"), "basis": "full_survivor",
                "clears": list(targets or []), "discloses": [],
                "grounds": "clears every fired anomaly; intent-preserving or "
                           "intent-revealing, so no requirement of the engineer's is altered"}
    adoptable = (rulings or {}).get("adoptable") or []
    if adoptable:
        best = min(adoptable, key=lambda r: (len(r["subset"]), r["subset"]))
        return {"ids": best["subset"], "tier": "T3", "basis": "authority_adopted",
                "clears": list(targets or []), "discloses": [],
                "grounds": best.get("model_grounds", ""),
                "intent_check": best.get("intent_check", [])}

    if not discharges:
        return None
    registry = registry or all_operations()
    by_id = {c["id"]: c for c in candidates}
    before = set(targets or [])

    #: Slots whose edits ADD representation or ADD a requirement. Only these may
    #: legitimately disclose new anomalies: promoting the analysis model or stating an
    #: omitted load case reveals inadequacy that was already there and unrepresentable.
    #: An edit to the geometry, the design variables or the constraints that creates a
    #: new anomaly has not disclosed anything -- it has made the design worse, and the
    #: guard below refuses it.
    DISCLOSING_SLOTS = ("load cases", "analysis model", "discretisation")

    best = None
    for d in discharges:
        if d.get("verdict") in ("INFEASIBLE", "NOT_ASSESSED", "NOT_REPRESENTABLE"):
            continue
        cleared = set(d.get("cleared") or [])
        if not cleared:
            continue
        tiers, slots = [], []
        for cid in d["ids"]:
            c = by_id.get(cid)
            op = registry.get(c["operation"]) if c else None
            tiers.append(getattr(op, "tier", "T3"))
            slots.append(getattr(op, "slot", "?").lower())
        if "T3" in tiers:
            continue                       # route 3 is closed to intent-altering edits
        disclosed = sorted(set(d.get("fired_after") or []) - before)
        may_disclose = all(any(k in s for k in DISCLOSING_SLOTS) for s in slots)
        if disclosed and not may_disclose:
            continue                       # a regression, not a disclosure
        ratio = float((d.get("summary") or {}).get("max_stress_ratio") or 1e9)
        key = (-len(cleared), len(disclosed), len(d["ids"]), round(ratio, 3), tuple(d["ids"]))
        if best is None or key < best[0]:
            best = (key, {"ids": list(d["ids"]),
                          "tier": max(tiers, default="T1"),
                          "basis": "disclosure_step",
                          "clears": sorted(cleared),
                          "discloses": disclosed,
                          "max_stress_ratio_after": ratio,
                          "grounds": ("clears " + ", ".join(sorted(cleared))
                                      + "; every edit in the set adds representation or "
                                        "states a requirement, so any newly visible "
                                        "anomaly was already present and unrepresentable "
                                        "rather than caused by this edit")})
    if best:
        return best[1]

    # Route 4: an intent-altering partial repair, ruled on by the authority stage.
    # Without this route the loop can only ever apply T1/T2 edits, and a problem whose
    # only real lever is intent-altering -- out-of-plane bending is governed by
    # thickness cubed, and thickness is intent-altering -- is unreachable no matter how
    # many epochs run. Observed: four epochs of marginal T1/T2 edits while the peak
    # stress rose, because the thickness edit was written, tiered T3, and then never
    # presented to the authority stage since it cleared some but not all anomalies.
    adoptable_partial = (rulings or {}).get("adoptable_partial") or []
    if adoptable_partial:
        best = min(adoptable_partial, key=lambda r: (len(r["subset"]), r["subset"]))
        return {"ids": best["subset"], "tier": "T3", "basis": "authority_adopted_partial",
                "clears": best.get("clears", []), "discloses": best.get("discloses", []),
                "grounds": best.get("model_grounds", ""),
                "intent_check": best.get("intent_check", [])}
    return None


def diagnose_once(spec, backend, intent, *, max_subset_size: int = 3,
                  discharge_max_iter: int | None = None, max_rounds: int = 2,
                  max_hypotheses: int = 6, active_predicates: tuple | None = None,
                  solve_cache: dict | None = None, discharge_n_cells: int | None = None,
                  tried_quantities=None, log=print) -> tuple:
    """One epoch of diagnosis. Returns ``(record, baseline_result)``."""
    active = active_predicates or ALL_PREDICATES
    solve_cache = solve_cache if solve_cache is not None else {}

    log("[1/7] convergence gate")
    baseline = solve_stress_problem(spec)
    summary = baseline.summary()
    log(f"      {summary['termination']}  iters={summary['iterations']}  "
        f"model={summary['model']}  cases={summary['load_cases']}  "
        f"peak={summary['max_von_mises_MPa']:.1f} MPa")
    if not baseline.converged:
        raise ConvergenceGateRefusal(
            f"terminated as {baseline.termination!r}; that is consistency repair, "
            "not adequacy diagnosis")

    log("[2/7] anomaly detection")
    report = detect(baseline, active=active, intent=intent)
    for a in report.anomalies:
        mark = "FIRED " if a.fired else ("      " if a.active else "[off] ")
        log(f"      {mark} {a.id}: {a.statement[:150]}")
    targets = report.fired_ids()
    record = {"spec": spec.to_dict(), "baseline": summary, "targets": targets,
              "anomalies": report.to_dict(), "design_state": _design_state(baseline)}
    if not targets:
        record.update({"outcome": "no_anomaly", "chosen": None})
        return record, baseline

    all_candidates, discharges, synthesis_records, rounds = [], [], [], []
    ranking, prior_evidence, gen, mm = None, None, {"hypotheses": []}, {"matched": [], "gaps": [], "rejected": []}

    for rnd in range(1, max_rounds + 1):
        log(f"  --- round {rnd} ---")
        log("[3a/7] blind hypothesis generation (no operation catalogue in the payload)")
        gen = propose_blind(backend, spec, report, summary, intent,
                            max_hypotheses=max_hypotheses, prior_evidence=prior_evidence)
        for h in gen["hypotheses"]:
            log(f"      {h['id']} [{h['tuple_slot']}] {h['claim'][:130]}")

        log("[3b/7] capability match against the committed catalogue")
        mm = capability_match(backend, gen["hypotheses"])
        for c in mm["matched"]:
            log(f"      MATCHED {c['id']} -> {c['operation']}({json.dumps(c['parameters'])})")
        for g in mm["gaps"]:
            log(f"      GAP     {g['hypothesis']['id']} -> {str(g['missing_capability'])[:110]}")

        log(f"[3c/7] tool synthesis for {len(mm['gaps'])} gap(s)")
        round_cands = list(mm["matched"])
        for g in mm["gaps"]:
            rec = synthesize(backend, g, spec, intent)
            synthesis_records.append(rec)
            if rec.get("admitted"):
                op = all_operations()[rec["operation"]]
                log(f"      WROTE   {rec['operation']}  slot={op.slot}  tier={rec['tier']}"
                    f"  (writer proposed {rec.get('proposed_tier')})")
                round_cands.append(rec["candidate"])
            else:
                log(f"      REJECTED {rec['hypothesis_id']}: {rec['rejected_because'][:130]}")

        # Deterministic coverage check before discharging: a spec-level target with
        # no candidate capable of addressing it means the matcher absorbed the
        # hypothesis into a tool that cannot realise it. Fill that gap by writing a
        # tool rather than reporting a no-survivor caused by the matcher.
        cov = uncovered_targets(spec, targets, round_cands, intent,
                                max_subset_size=max_subset_size)
        for g in cov:
            log(f"      UNCOVERED {g['hypothesis']['targets_anomaly']} -> forcing synthesis")
            rec = synthesize(backend, g, spec, intent)
            synthesis_records.append(rec)
            if rec.get("admitted"):
                op = all_operations()[rec["operation"]]
                log(f"      WROTE   {rec['operation']}  slot={op.slot}  tier={rec['tier']}"
                    f"  (coverage-forced)")
                round_cands.append(rec["candidate"])
            else:
                log(f"      REJECTED coverage synthesis: {rec['rejected_because'][:130]}")

        seen = {c["id"] for c in all_candidates}
        for c in round_cands:
            while c["id"] in seen:
                c["id"] += "b"
            seen.add(c["id"])
            all_candidates.append(c)
        if not all_candidates:
            rounds.append({"round": rnd, "outcome": "no_candidates"})
            continue

        log(f"[4/7] deductive discharge, subsets up to size {max_subset_size}")

        def progress(i, n, d):
            extra = f"  cleared={d.cleared}" if d.cleared and d.verdict != "SURVIVES" else ""
            log(f"      {i:>3}/{n}  {'+'.join(d.ids):<20} {d.verdict}{extra}")

        discharges = _discharge(spec, all_candidates, targets,
                                max_subset_size=max_subset_size,
                                max_iter=discharge_max_iter, active=active,
                                cache=solve_cache, progress=progress, intent=intent,
                                n_cells=discharge_n_cells)
        log("[5/7] ranking")
        ranking = rank(discharges, all_candidates)
        log(f"      {ranking['outcome']}: {ranking['statement'][:200]}")
        rounds.append({"round": rnd, "outcome": ranking["outcome"],
                       "counts": ranking["counts"],
                       "candidates": [c["id"] for c in all_candidates]})

        prior_evidence = {
            "previous_round": rnd, "outcome": ranking["outcome"],
            "counts": ranking["counts"],
            "discharges": [{"ids": list(d.ids), "edits": d.operations,
                            "verdict": d.verdict, "cleared": list(d.cleared),
                            "remaining": list(d.remaining), "reason": d.reason[:300]}
                           for d in discharges],
            "synthesized_this_run": sorted(SYNTHESIZED),
            "rejected_synthesis": [{"hypothesis_id": r["hypothesis_id"],
                                    "why": r.get("rejected_because", "")}
                                   for r in synthesis_records if not r.get("admitted")],
        }
        if ranking["outcome"] != "no_survivor":
            break

    log("[6/7] authority: ruling on intent-altering survivors against declared intent")
    rulings = authority_rule(backend, ranking or {}, all_candidates, intent,
                             discharges=[d.to_dict() for d in discharges], targets=targets)
    for r in rulings.get("rulings", []):
        log(f"      {'+'.join(r['subset']):<20} model={r['model_ruling']:<9} "
            f"final={r['ruling']:<9} intent_permits={r['intent_permits_adoption']}")
        if r.get("override"):
            log(f"          OVERRIDE: {r['override']}")

    chosen = pick_edit_set(ranking, rulings, all_candidates,
                           discharges=[d.to_dict() for d in discharges], targets=targets)
    log(f"[7/7] accept: {('+'.join(chosen['ids']) + ' via ' + chosen['basis']) if chosen else 'nothing authorised'}")

    record.update({
        "outcome": (ranking or {}).get("outcome", "no_candidates"),
        "rounds": rounds, "hypotheses": gen["hypotheses"],
        "capability_match": {
            "matched": [{k: v for k, v in c.items() if k != "hypothesis"} for c in mm["matched"]],
            "gaps": [{"hypothesis_id": g["hypothesis"]["id"],
                      "missing_capability": g["missing_capability"],
                      "note": g.get("note", "")} for g in mm["gaps"]],
            "rejected": mm["rejected"]},
        "synthesis": synthesis_records,
        "candidates": [{k: v for k, v in c.items() if k != "hypothesis"} for c in all_candidates],
        "discharges": [d.to_dict() for d in discharges],
        "ranking": ranking, "authority": rulings, "chosen": chosen,
    })
    return record, baseline


def run_iterative_cascade(base_spec, backend, intent=None, *, max_epochs: int = 3,
                          max_subset_size: int = 3, discharge_max_iter: int | None = None,
                          max_rounds: int = 2, max_hypotheses: int = 6,
                          accept_max_iter: int | None = None,
                          active_predicates: tuple | None = None,
                          discharge_n_cells: int | None = None,
                          verbose: bool = True) -> dict:
    """Diagnose, apply what is authorised, re-diagnose, until quiet or stuck.

    ``discharge_n_cells`` coarsens the re-solve inside the discharge stage only.
    Baseline solves, accepted design states, and every number reported as a result
    stay at the specification's own discretisation; only the verdicts are cheaper.
    The value used is recorded in the returned record so a reader can see which
    verdicts rest on a coarser mesh than the designs they are about.
    """
    t0 = time.time()
    log = print if verbose else (lambda *a, **k: None)
    solve_cache: dict = {}
    spec = base_spec.copy()
    epochs, states, applied = [], [], []
    tried_quantities: list = []

    for ep in range(1, max_epochs + 1):
        log(f"\n{'=' * 72}\nEPOCH {ep}\n{'=' * 72}")
        rec, baseline = diagnose_once(
            spec, backend, intent, max_subset_size=max_subset_size,
            discharge_max_iter=discharge_max_iter, max_rounds=max_rounds,
            max_hypotheses=max_hypotheses, active_predicates=active_predicates,
            solve_cache=solve_cache, discharge_n_cells=discharge_n_cells,
            tried_quantities=tried_quantities, log=log)
        rec["epoch"] = ep
        states.append({"epoch": ep, "label": ("stated problem" if ep == 1
                                              else "after " + " + ".join(applied[-1]["operations"])),
                       "result": baseline, **rec["design_state"],
                       "targets_fired": rec["targets"]})
        epochs.append(rec)
        for h in rec.get("hypotheses", []):
            tried_quantities.append(str(h.get("quantity", "")))
        for c in rec.get("candidates", []):
            tried_quantities.extend(str(k) for k in (c.get("parameters") or {}))

        if not rec["targets"]:
            log("      every active predicate is quiet: the specification is adequate "
                "as far as these predicates can tell")
            break
        chosen = rec.get("chosen")
        if not chosen:
            log("      nothing further is authorised; stopping with an unresolved diagnosis")
            break

        by_id = {c["id"]: c for c in rec["candidates"]}
        edits = [by_id[i] for i in chosen["ids"] if i in by_id]
        spec = apply_edits(spec, edits)
        if accept_max_iter:
            spec.optimizer.max_iter = max(spec.optimizer.max_iter, accept_max_iter)
        applied.append({"epoch": ep, "ids": chosen["ids"],
                        "clears": chosen.get("clears", []),
                        "discloses": chosen.get("discloses", []),
                        "operations": [e["operation"] for e in edits],
                        "parameters": {e["operation"]: e.get("parameters") for e in edits},
                        "tier": chosen.get("tier"), "basis": chosen["basis"],
                        "grounds": chosen.get("grounds", "")})
        log(f"      applied {[e['operation'] for e in edits]} via {chosen['basis']}"
            f"; clears={chosen.get('clears')} discloses={chosen.get('discloses')}; re-diagnosing")

    final_targets = epochs[-1]["targets"] if epochs else []
    last = states[-1]
    resolved = (not final_targets) and last["within_allowable_by_case"] and \
        all(last["within_allowable_by_case"].values())
    return {
        "resolved": bool(resolved),
        "epochs_run": len(epochs),
        "discharge_discretisation": {
            "n_cells_per_side": discharge_n_cells or base_spec.geometry.n_cells_per_side,
            "baseline_n_cells_per_side": base_spec.geometry.n_cells_per_side,
            "coarsened": bool(discharge_n_cells
                              and discharge_n_cells != base_spec.geometry.n_cells_per_side),
            "note": ("Verdicts are established at this in-plane discretisation. Baseline "
                     "solves and every accepted design state use the specification's own "
                     "value. Where these differ, a verdict is a weaker claim than the "
                     "design it concerns."),
        },
        "declared_intent": intent.to_dict() if intent is not None else None,
        "applied_edit_sets": applied,
        "final_spec": spec.to_dict(),
        "final_state": {k: v for k, v in last.items() if k != "result"},
        "design_states": [{k: v for k, v in s.items() if k != "result"} for s in states],
        "epochs": epochs,
        "synthesized_operations": {n: {"slot": o.slot, "tier": o.tier,
                                       "boundedness": o.boundedness,
                                       "parameter_source": o.parameter_source,
                                       "description": o.description}
                                   for n, o in SYNTHESIZED.items()},
        "wall_seconds": time.time() - t0,
        "_results": [s["result"] for s in states],
    }

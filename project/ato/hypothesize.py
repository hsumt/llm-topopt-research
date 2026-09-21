"""Stage 3: a language model proposes candidate specification edits.

The model sees the specification, the deterministic anomaly evidence and the
edit grammar.  It returns candidates only.  It does not decide survival, it does
not decide admissibility tier, and it is told so explicitly -- those come from
re-solving and from ``grammar.py``.

Malformed candidates are not discarded silently: the validation errors are fed
back and the model is asked once to repair them, which keeps generation honest
without letting an invalid edit through.
"""
from __future__ import annotations

import json

from project.ato.grammar import OPERATIONS, grammar_for_prompt, validate_candidate

SYSTEM_PROMPT = """You are the hypothesis generation stage of a specification-adequacy \
diagnosis system for topology optimization.

A structural optimization run has CONVERGED CORRECTLY on a FEASIBLE stated problem. \
The solver did not fail. The optimality conditions hold for the problem as posed. \
Deterministic predicates have nonetheless detected an anomaly in the converged design. \
The working hypothesis is therefore that the STATED PROBLEM was itself inadequate: some \
slot of the problem statement is wrong, missing, or under-specified.

Your job is to propose candidate specification edits that could explain the anomaly. \
Each candidate must be a single operation drawn from the supplied edit grammar, with \
concrete parameter values.

Hard constraints on your output:
- Propose ONLY operations that appear in the supplied grammar, with exactly its parameter names.
- Do NOT state whether a candidate will succeed, survive, or fix the anomaly. Every candidate \
is applied and the modified problem is re-solved; the re-solve decides, not you.
- Do NOT invent metrics, and do not assert values that are not in the evidence packet.
- Propose candidates that are genuinely distinct explanations, including ones you expect to \
be eliminated. A candidate that is eliminated by re-solving is a useful result, not a wasted one.
- Include at least one candidate that would change what the engineer asked for, if such an \
edit is plausible, so that the authority gate has something to rule on.
- Choose parameter values that are defensible from the evidence and the geometry, not round \
numbers chosen for convenience.

You may propose an admissibility tier, but it is advisory only and will be overridden by the \
grammar's own classification."""

CANDIDATE_TOOL = {
    "name": "propose_candidates",
    "description": "Return candidate specification edits that could explain the anomaly.",
    "input_schema": {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": 3,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string",
                               "description": "short stable identifier, e.g. 'C1'"},
                        "operation": {"type": "string",
                                      "description": "an operation name from the grammar"},
                        "parameters": {"type": "object",
                                       "description": "parameter values for that operation"},
                        "targets_anomaly": {
                            "type": "array", "items": {"type": "string"},
                            "description": "which fired anomaly ids this edit is meant to explain"},
                        "proposed_tier": {"type": "string", "enum": ["T1", "T2", "T3"]},
                        "rationale": {
                            "type": "string",
                            "description": "why this slot of the problem statement could be the defect"},
                    },
                    "required": ["id", "operation", "parameters", "targets_anomaly", "rationale"],
                },
            }
        },
        "required": ["candidates"],
    },
}


def _evidence_packet(spec, report, run_summary) -> str:
    return json.dumps(
        {
            "specification": spec.to_dict(),
            "run_outcome": run_summary,
            "anomaly_evidence": report.to_dict(),
            "edit_grammar": grammar_for_prompt(),
        },
        indent=2,
    )


def propose(backend, spec, report, run_summary, *, max_candidates: int = 6) -> dict:
    """Generate and validate candidate edits. Returns {candidates, rejected, raw, usage}."""
    user = (
        _evidence_packet(spec, report, run_summary)
        + "\n\nPropose candidate specification edits that could explain the fired anomalies. "
        + f"Return at most {max_candidates}."
    )
    raw = backend(SYSTEM_PROMPT, user, CANDIDATE_TOOL)
    cands = list(raw.get("candidates", []))

    good, bad = [], []
    for c in cands:
        problems = validate_candidate(c)
        (good if not problems else bad).append(
            c if not problems else {"candidate": c, "problems": problems}
        )

    if bad:
        repair = (
            user
            + "\n\nYour previous response contained candidates that failed grammar validation:\n"
            + json.dumps(bad, indent=2)
            + "\n\nReturn a corrected set. Keep the candidates that were valid, repair the rest, "
              "and drop any that cannot be expressed in the grammar."
        )
        raw2 = backend(SYSTEM_PROMPT, repair, CANDIDATE_TOOL)
        good, bad = [], []
        for c in raw2.get("candidates", []):
            problems = validate_candidate(c)
            (good if not problems else bad).append(
                c if not problems else {"candidate": c, "problems": problems}
            )
        raw = raw2

    # De-duplicate on (operation, parameters) and stabilise ids.
    seen, unique = set(), []
    for c in good:
        key = (c["operation"], json.dumps(c["parameters"], sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)

    for c in unique:
        op = OPERATIONS[c["operation"]]
        c["tier"] = op.tier
        c["tier_source"] = "grammar"
        c["tier_agreed_with_model"] = (c.get("proposed_tier") == op.tier)
        c["slot"] = op.slot
        c["boundedness"] = op.boundedness
        c["parameter_source"] = op.parameter_source

    return {
        "candidates": unique[:max_candidates],
        "rejected": bad,
        "usage": dict(getattr(backend, "usage", {})),
        "model": getattr(backend, "model", "unknown"),
    }


NEXT_ROUND_NOTE = """A previous round of candidates has already been discharged by re-solving. \
The outcomes are given below. Read them as hard evidence about the problem, not as feedback on \
style:

- ELIMINATED means the edit was applied, the problem re-solved, and the anomaly still fired. \
That explanation is ruled out on its own.
- INFEASIBLE means the edited specification admits no feasible design at all. This is strong \
evidence: the edited requirement set is jointly unsatisfiable, so some *other* stated requirement \
must give way for that edit to be adoptable. Consider what would have to be relaxed alongside it, \
including requirements the engineer did state.
- NOT_ASSESSED means the re-solve produced no trustworthy verdict; no conclusion follows.
- 'cleared' lists anomalies the edit did remove even where the overall verdict was not survival. \
An edit that clears part of the anomaly set is a component of a coupled explanation.

Propose a NEW set of candidates informed by these outcomes. Do not simply repeat an eliminated \
candidate with a cosmetically different parameter. Where the evidence points at a coupled defect \
spanning more than one slot of the problem statement, propose the components that would have to \
be combined, including any intent-altering edit that the evidence implies -- the authority gate \
exists to rule on exactly that, and withholding it would hide the finding."""


def propose_next_round(backend, spec, report, run_summary, prior_discharges,
                       *, max_candidates: int = 5) -> dict:
    """Generate a further round of candidates given the previous round's verdicts."""
    outcomes = [
        {"ids": d["ids"], "operations": d["operations"], "verdict": d["verdict"],
         "reason": d["reason"], "cleared": d.get("cleared", []),
         "still_firing": d.get("remaining", [])}
        for d in prior_discharges
    ]
    user = (
        _evidence_packet(spec, report, run_summary)
        + "\n\nPREVIOUS ROUND OUTCOMES:\n"
        + json.dumps(outcomes, indent=2)
        + "\n\n" + NEXT_ROUND_NOTE
        + f"\n\nReturn at most {max_candidates} candidates."
    )
    raw = backend(SYSTEM_PROMPT, user, CANDIDATE_TOOL)
    good, bad = [], []
    for c in raw.get("candidates", []):
        problems = validate_candidate(c)
        (good if not problems else bad).append(
            c if not problems else {"candidate": c, "problems": problems})
    for c in good:
        op = OPERATIONS[c["operation"]]
        c["tier"] = op.tier
        c["tier_source"] = "grammar"
        c["tier_agreed_with_model"] = (c.get("proposed_tier") == op.tier)
        c["slot"] = op.slot
        c["boundedness"] = op.boundedness
        c["parameter_source"] = op.parameter_source
    return {"candidates": good[:max_candidates], "rejected": bad,
            "usage": dict(getattr(backend, "usage", {})),
            "model": getattr(backend, "model", "unknown")}

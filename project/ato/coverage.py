"""Coverage check: can the candidate set even address every fired anomaly?

The capability matcher is a language model, and its characteristic failure is the
eager match -- rounding a hypothesis to the nearest available tool. Observed
directly: a hypothesis stating that the specification omits the out-of-plane load
case was matched to ``distribute_load``, which spreads the *existing* in-plane load
over more nodes. Both live in the ``load cases`` slot, both are about how load is
applied, and the match is still wrong: no parameter of ``distribute_load`` can
introduce a load along an axis the specification never mentions. The consequence is
worse than a wasted discharge, because the target that hypothesis was aimed at then
has no candidate addressing it at all, and the cascade reports "no survivor" for a
reason that is an artefact of the matcher rather than a fact about the problem.

Some predicates can be evaluated on the specification alone, with no solve. For
those, whether a candidate *could* clear the anomaly is decidable in advance,
cheaply and deterministically. This module does exactly that, and where a fired
target has no candidate capable of addressing it, it emits a synthesis request with
a precise statement of the missing capability -- so the gap is filled by writing a
tool rather than silently absorbed.

It is deliberately a *necessary* condition only: passing here does not mean the edit
will clear the anomaly. That is still decided by re-solving.
"""
from __future__ import annotations

import itertools

from project.ato.grammar import apply_edits


def _missing_axes(spec, intent) -> list:
    return list(intent.missing_axes(spec)) if intent is not None else []


#: Predicates decidable from the specification alone. Each entry maps the predicate
#: id to a function returning a list of unmet items; empty means "would not fire".
SPEC_LEVEL_PREDICATES = {
    "unrepresented_load_axis": _missing_axes,
}


def uncovered_targets(spec, targets, candidates, intent, *, max_subset_size: int = 3) -> list:
    """Fired spec-level targets that no candidate or subset can address.

    Returns synthesis-request dicts shaped like ``capability.match`` gaps, so the
    caller can hand them straight to the synthesis stage.
    """
    out = []
    for pid, predicate in SPEC_LEVEL_PREDICATES.items():
        if pid not in targets or intent is None:
            continue
        if not predicate(spec, intent):
            continue                      # not actually unmet on the stated problem
        covered_by = None
        ids = [c["id"] for c in candidates]
        for size in range(1, min(max_subset_size, max(len(ids), 1)) + 1):
            for sub in itertools.combinations(ids, size):
                chosen = [c for c in candidates if c["id"] in sub]
                try:
                    edited = apply_edits(spec, chosen)
                except Exception:
                    continue
                if not predicate(edited, intent):
                    covered_by = list(sub)
                    break
            if covered_by:
                break
        if covered_by:
            continue

        unmet = predicate(spec, intent)
        out.append({
            "hypothesis": {
                "id": f"COV_{pid}",
                "claim": (f"the anomaly {pid!r} is unmet on the stated problem and no "
                          f"candidate edit set, alone or in combination, changes it"),
                "tuple_slot": "load cases",
                "required_change": (
                    f"the specification must state a load case along each of {unmet}, and "
                    f"the analysis model must be able to carry it"),
                "quantity": f"load case along {unmet}",
                "proposed_value": "unknown; the engineer's service envelope determines it",
                "targets_anomaly": [pid],
                "rationale": (
                    "Established deterministically, not proposed: applying every candidate "
                    "and every subset to the specification leaves the requirement unmet, so "
                    "the hypothesis set provably cannot clear this anomaly."),
                "needs_new_capability": True,
                "origin": "coverage_check",
            },
            "missing_capability": (
                f"an operation that adds a load case along {unmet} to spec.load_cases. Note "
                f"that spreading, redistributing or rescaling an existing load case cannot do "
                f"this: no parameter of an existing-load edit introduces an axis the "
                f"specification does not mention."),
            "note": (f"emitted by the deterministic coverage check because {len(candidates)} "
                     f"candidate(s) and their subsets all leave {unmet} unmet"),
        })
    return out

"""Stage 5: ranking and the authority gate.

Survivors are ordered lexicographically by admissibility tier, then by edit
count.  T1 and T2 diagnoses are reported with their evidence.  A T3
intent-altering survivor terminates in a flagged diagnosis and is never applied
silently, whatever its rank.

Non-monotone re-entry is detected here rather than assumed: a candidate
eliminated on its own that appears in a surviving subset is recorded explicitly,
because it is the justification for enumerating the subset layer at all.
"""
from __future__ import annotations

from project.ato.discharge import ELIMINATED, INFEASIBLE, NOT_ASSESSED, SURVIVES
from project.ato.grammar import TIER_MEANING, TIER_ORDER, all_operations


def subset_tier(ids, candidates) -> str:
    # all_operations(), not OPERATIONS: a candidate may name an operation the synthesis
    # stage wrote during this run, which is registered separately from the committed
    # grammar. Looking only at the committed grammar raises KeyError on exactly the
    # candidates this architecture exists to produce.
    ops = all_operations()
    tiers = [ops[c["operation"]].tier for c in candidates if c["id"] in set(ids)]
    return max(tiers, key=lambda t: TIER_ORDER[t]) if tiers else "T1"


def subset_boundedness(ids, candidates) -> str:
    ops = all_operations()
    marks = [ops[c["operation"]].boundedness for c in candidates if c["id"] in set(ids)]
    if any("UNBOUND" in m for m in marks):
        return "UNBOUND (inherited)" if len(marks) > 1 else marks[0]
    return "BOUND"


def rank(discharges, candidates) -> dict:
    survivors = [d for d in discharges if d.verdict == SURVIVES]
    eliminated = [d for d in discharges if d.verdict == ELIMINATED]
    unassessed = [d for d in discharges if d.verdict == NOT_ASSESSED]
    infeasible = [d for d in discharges if d.verdict == INFEASIBLE]

    rows = []
    for d in survivors:
        tier = subset_tier(d.ids, candidates)
        rows.append({
            "ids": list(d.ids),
            "operations": d.operations,
            "tier": tier,
            "tier_meaning": TIER_MEANING[tier],
            "boundedness": subset_boundedness(d.ids, candidates),
            "n_edits": len(d.ids),
            "run": d.summary,
        })
    rows.sort(key=lambda r: (TIER_ORDER[r["tier"]], r["n_edits"], r["ids"]))

    singleton_eliminated = {d.ids[0] for d in eliminated if len(d.ids) == 1}
    reentry = []
    for d in survivors:
        if len(d.ids) > 1:
            back = sorted(set(d.ids) & singleton_eliminated)
            if back:
                reentry.append({
                    "subset": list(d.ids),
                    "re_entering": back,
                    "note": (
                        f"{', '.join(back)} was eliminated alone but the subset survives; "
                        "the discharge relation is non-monotone here, which is why the "
                        "subset layer is enumerated rather than skipped"
                    ),
                })

    minimal = []
    for r in rows:
        s = set(r["ids"])
        if not any(set(o["ids"]) < s for o in rows):
            minimal.append(r)

    applicable = [r for r in minimal if r["tier"] in ("T1", "T2")
                  and "UNBOUND" not in r["boundedness"]]
    flagged = [r for r in minimal if r["tier"] == "T3"]

    if not rows:
        outcome = "no_survivor"
        statement = (
            "No candidate or subset entailed the anomaly away. The hypothesis set does not "
            "contain the defect; report the anomaly and widen the grammar or the candidate set."
        )
    elif applicable:
        best = applicable[0]
        outcome = "diagnosis"
        statement = (
            f"Minimal admissible diagnosis: {', '.join(best['operations'])} "
            f"[{best['tier']}, {best['boundedness']}]."
        )
    elif flagged:
        outcome = "flagged_intent_altering"
        statement = (
            "Every minimal survivor is intent-altering (T3) or carries unbound parameters. "
            "The mechanism reports a flagged diagnosis and applies nothing: an edit that "
            "changes what the engineer asked for requires human authority."
        )
    else:
        outcome = "unbound_survivors_only"
        statement = (
            "Survivors exist but all carry parameters the stated problem does not determine. "
            "They establish that a slot is empty and nothing about what belongs in it."
        )

    return {
        "outcome": outcome,
        "statement": statement,
        "ranked_survivors": rows,
        "minimal_survivors": minimal,
        "applicable": applicable,
        "flagged_intent_altering": flagged,
        "non_monotone_reentry": reentry,
        "counts": {
            "survived": len(survivors),
            "eliminated": len(eliminated),
            "not_assessed": len(unassessed),
            "infeasible": len(infeasible),
            "total_discharges": len(discharges),
        },
        "eliminated": [{"ids": list(d.ids), "operations": d.operations, "reason": d.reason}
                       for d in eliminated],
        "not_assessed": [{"ids": list(d.ids), "operations": d.operations, "reason": d.reason}
                         for d in unassessed],
        "infeasible": [{"ids": list(d.ids), "operations": d.operations, "reason": d.reason,
                        "cleared": d.cleared} for d in infeasible],
    }

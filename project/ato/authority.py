"""The authority stage: ruling on intent-altering edits.

A T3 edit asserts something the engineer never stated. Applying one on the
system's own initiative is the failure mode the whole tier scheme exists to
prevent, so the question is not "may the system decide?" but "from where does the
authority to decide come?"

It comes from the written intent document, and from nowhere else. If the engineer
has declared in ``intent.py`` that the thickness is a free design choice, then
changing the thickness is inside delegated authority -- the engineer has already
spoken, and the system is applying their statement rather than substituting its
own judgement. If the intent is silent on a quantity, or declares it fixed, no
amount of reasoning about how sensible the edit looks confers authority to make
it, and the ruling must escalate to a human.

That division is enforced structurally, not by prompt discipline:

* the model reasons about grounds and produces a recommendation;
* ``_permitted_by_intent`` decides independently, from the declared document,
  whether adoption is *available* at all;
* a recommendation to adopt something the intent does not cover is downgraded to
  ``escalate`` and the disagreement is recorded.

So the model can never widen its own authority, and a run where it tried is
visible in the record.
"""
from __future__ import annotations

import json

RULINGS = ("adopt", "reject", "escalate")

AUTHORITY_SYSTEM_PROMPT = """You are the engineering-authority stage of a specification-adequacy \
diagnosis system. You rule on candidate edits that would ALTER what the engineer asked for.

You have been given the declared design intent: the engineer's written statement of what the \
part must do and which quantities are theirs to fix or free. Rule each edit against that \
document, and against nothing else.

  adopt    -- the declared intent explicitly authorises this class of change (the quantity is \
listed as free to change, or the limit it moves is declared negotiable) AND the edit is \
consistent with every other stated requirement. Quote the part of the intent that authorises it.

  reject   -- the declared intent contradicts the edit: it moves a quantity declared fixed, or \
relaxes a limit declared firm. Quote the contradiction.

  escalate -- the intent is SILENT on this quantity. Not "probably fine", not "any engineer \
would accept it". Silence means the engineer has not spoken and you must not speak for them.

RULES

1. Engineering plausibility is NOT authority. An edit can be obviously sensible and still \
require escalation, because the question is who is entitled to decide, not what the right \
answer is.

2. Quote the intent. A ruling whose grounds are not traceable to the declared document is \
worthless for audit, and will be recorded as unsupported.

3. You are ruling on ADMISSIBILITY, not on correctness. Whether the edit removes the anomaly \
has already been established by re-solving; do not re-litigate it.

4. If an edit clears the anomaly only in combination with others, rule on the combination as \
presented and say so."""

AUTHORITY_TOOL = {
    "name": "rule_on_edits",
    "description": "Rule on intent-altering candidate edits against the declared design intent.",
    "input_schema": {
        "type": "object",
        "properties": {
            "rulings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "subset": {"type": "array", "items": {"type": "string"},
                                   "description": "candidate ids this ruling covers"},
                        "ruling": {"type": "string", "enum": ["adopt", "reject", "escalate"]},
                        "grounds": {"type": "string",
                                    "description": "quote or close paraphrase of the intent clause relied on"},
                        "residual_risk": {"type": "string",
                                          "description": "what remains unresolved even if adopted"},
                        "question_for_engineer": {
                            "type": "string",
                            "description": "when escalating: the single question that would settle it, "
                                           "answerable from intent rather than from numbers"},
                    },
                    "required": ["subset", "ruling", "grounds"],
                },
            }
        },
        "required": ["rulings"],
    },
}


def _quantity_of(op_name: str, slot: str) -> str:
    """Best-effort name of the quantity an edit moves, for the intent lookup."""
    s = (op_name + " " + slot).lower()
    for q in ("thickness", "fillet", "mass", "stress", "load", "mesh", "filter",
              "cluster", "model"):
        if q in s:
            return q
    return slot.lower()


def _permitted_by_intent(op, intent) -> tuple:
    """Independent check: does the declared document authorise adopting this edit?

    Deliberately narrow and deliberately not delegated. Returns
    ``(permitted, grounds)``.
    """
    if intent is None:
        return False, "no design intent document was supplied, so nothing is authorised"
    q = _quantity_of(op.name, op.slot)
    if q in {str(x).lower() for x in intent.fixed_geometry}:
        return False, f"the intent declares {q!r} fixed"
    if q in {str(x).lower() for x in intent.mutable_geometry}:
        return True, f"the intent lists {q!r} among the quantities the engineer permits changing"
    if "stress" in q and not intent.stress_allowable_is_firm:
        return True, "the intent declares the stated stress allowable negotiable"
    if "mass" in q and not intent.mass_budget_is_firm:
        return True, "the intent declares the stated mass budget negotiable"
    if "stress" in q and intent.stress_allowable_is_firm:
        return False, "the intent declares the stated stress allowable firm"
    if "mass" in q and intent.mass_budget_is_firm:
        return False, "the intent declares the stated mass budget firm"
    return False, f"the intent is silent on {q!r}"


def _partial_repairs(discharges, candidates, targets, registry) -> list:
    """Intent-altering edit sets that clear SOME of the fired anomalies.

    These have to reach the authority stage. A survivor-only authority stage can
    only ever authorise an edit that fixes everything at once, so a problem whose
    dominant lever is intent-altering stays permanently out of reach: observed on
    the L-bracket, where out-of-plane bending is governed by the cube of the
    thickness, thickness is intent-altering, and four epochs of intent-preserving
    edits ran while the peak stress rose. The engineer's document already says
    whether that lever is theirs to pull; refusing to ask is not caution, it is a
    guaranteed non-answer.
    """
    by_id = {c["id"]: c for c in candidates}
    before = set(targets or [])
    rows = []
    for d in discharges or []:
        if d.get("verdict") in ("INFEASIBLE", "NOT_ASSESSED", "NOT_REPRESENTABLE"):
            continue
        cleared = sorted(set(d.get("cleared") or []))
        if not cleared or set(cleared) >= before:
            continue                      # nothing cleared, or a full survivor
        tiers = []
        for cid in d["ids"]:
            c = by_id.get(cid)
            op = registry.get(c["operation"]) if c else None
            tiers.append(getattr(op, "tier", "T3"))
        if "T3" not in tiers:
            continue                      # T1/T2 partials need no authority
        rows.append({"ids": list(d["ids"]), "cleared": cleared,
                     "remaining": sorted(set(d.get("remaining") or [])),
                     "disclosed": sorted(set(d.get("fired_after") or []) - before),
                     "max_stress_ratio_after": (d.get("summary") or {}).get("max_stress_ratio"),
                     "partial": True})
    rows.sort(key=lambda r: (-len(r["cleared"]), len(r["ids"])))
    return rows[:6]


def rule(backend, ranking: dict, candidates: list, intent=None, *, registry=None,
         discharges=None, targets=None) -> dict:
    """Rule on intent-altering survivors and partial repairs alike."""
    from project.ato.grammar import all_operations

    registry = registry or all_operations()
    by_id = {c["id"]: c for c in candidates}
    flagged = list(ranking.get("flagged_intent_altering") or [])
    partials = _partial_repairs(discharges, candidates, targets, registry)
    rows_in = [dict(r, partial=False) for r in flagged] + partials
    if not rows_in:
        return {"rulings": [], "adoptable": [], "adoptable_partial": [],
                "escalated": [], "rejected": [],
                "note": "no intent-altering survivor or partial repair to rule on"}

    presented = []
    for row in rows_in:
        ops = []
        for cid in row["ids"]:
            c = by_id.get(cid)
            if not c:
                continue
            op = registry.get(c["operation"])
            ops.append({"candidate_id": cid, "operation": c["operation"],
                        "parameters": c.get("parameters"),
                        "slot": getattr(op, "slot", "?"),
                        "tier": getattr(op, "tier", "?"),
                        "boundedness": getattr(op, "boundedness", "?"),
                        "origin": c.get("origin", "committed_grammar"),
                        "rationale": c.get("rationale", "")})
        presented.append({"subset": row["ids"], "edits": ops,
                          "clears": row.get("cleared", row.get("clears", [])),
                          "leaves_unresolved": row.get("remaining", []),
                          "newly_disclosed": row.get("disclosed", []),
                          "peak_stress_ratio_after": row.get("max_stress_ratio_after"),
                          "clears_every_anomaly": not row.get("partial", False)})

    user = (
        intent.summary_for_prompt() + "\n\n" if intent is not None else ""
    ) + (
        "INTENT-ALTERING EDIT SETS, each applied and re-solved. Those with "
        "clears_every_anomaly=true remove every fired anomaly; the rest are PARTIAL "
        "repairs that remove some and leave others, which is still worth authorising if "
        "the intent permits the change -- an unresolved anomaly is not a reason to refuse "
        "a change the engineer has already sanctioned:\n"
        + json.dumps(presented, indent=2, default=str)
    )
    raw = backend(AUTHORITY_SYSTEM_PROMPT, user, AUTHORITY_TOOL)

    partial_ids = {tuple(sorted(r["ids"])) for r in partials}
    rulings, adoptable, adoptable_partial, escalated, rejected = [], [], [], [], []
    for r in raw.get("rulings", []):
        subset = list(r.get("subset") or [])
        model_ruling = r.get("ruling")
        permitted, grounds = True, []
        for cid in subset:
            c = by_id.get(cid)
            op = registry.get(c["operation"]) if c else None
            if op is None:
                permitted = False
                grounds.append(f"{cid}: unknown operation")
                continue
            ok, why = _permitted_by_intent(op, intent)
            grounds.append(f"{op.name}: {why}")
            permitted = permitted and ok
        final = model_ruling
        override = None
        if model_ruling == "adopt" and not permitted:
            final = "escalate"
            override = ("recommended adoption, but the declared intent does not authorise "
                        "every edit in the set; downgraded to escalate")
        row = {"subset": subset, "model_ruling": model_ruling, "ruling": final,
               "model_grounds": r.get("grounds", ""),
               "intent_check": grounds,
               "intent_permits_adoption": permitted,
               "override": override,
               "residual_risk": r.get("residual_risk", ""),
               "question_for_engineer": r.get("question_for_engineer", "")}
        is_partial = tuple(sorted(subset)) in partial_ids
        src = next((r for r in rows_in if sorted(r["ids"]) == sorted(subset)), {})
        row.update({"partial": is_partial, "clears": src.get("cleared", []),
                    "discloses": src.get("disclosed", [])})
        rulings.append(row)
        if final == "adopt":
            (adoptable_partial if is_partial else adoptable).append(row)
        elif final == "reject":
            rejected.append(row)
        else:
            escalated.append(row)

    return {"rulings": rulings, "adoptable": adoptable,
            "adoptable_partial": adoptable_partial, "escalated": escalated,
            "rejected": rejected, "model": getattr(backend, "model", "unknown")}

"""Blind hypothesis generation: propose what is wrong, not which tool to use.

The earlier design handed the model the edit grammar and asked it to pick an
operation. That biases the search toward whatever the toolbox already contains: if
every available operation sets a scalar on an existing field, every hypothesis
looks like parameter tuning, and a defect that needs a capability the software
lacks can never even be voiced.

This stage is therefore **blind to the toolbox**. The model sees the specification,
the deterministic anomaly evidence, the declared design intent, and the vocabulary
of the canonical problem tuple -- which is theory, not tooling. It does not see the
operation catalogue, and the tool schema has no operation field to fill. It states,
in its own terms, what about the stated problem would have to be different.

Matching a hypothesis to an available capability happens afterwards in
``capability.py``, and where no capability exists ``synthesize.py`` writes one.
Keeping those stages separate is what makes "the toolbox was missing something" an
observable outcome rather than an invisible ceiling.
"""
from __future__ import annotations

BLIND_SYSTEM_PROMPT = """You are the hypothesis stage of a specification-adequacy diagnosis \
system for structural optimization.

A structural optimization run has CONVERGED CORRECTLY on a FEASIBLE stated problem. The \
solver did not fail; the optimality conditions hold for the problem as posed. Deterministic \
predicates have nonetheless found something wrong with the converged design, or with the \
completeness of the problem statement itself. The working hypothesis is that the STATED \
PROBLEM was inadequate.

Your job is to say WHAT ABOUT THE STATED PROBLEM WOULD HAVE TO BE DIFFERENT. Nothing more.

Frame every hypothesis against one slot of the canonical problem statement:
  - objective                     what is being minimised
  - design variables              what the optimiser is allowed to change
  - design domain                 the geometry, including regions excluded from design
  - load cases                    every load the part must carry, and in which direction
  - boundary conditions           how the part is held
  - inequality constraints        stated limits: stress allowables, mass budgets
  - discretisation / analysis model   how the physics is represented at all

CRITICAL INSTRUCTIONS

1. You are deliberately NOT told what edits the software can perform. Do not guess, do not \
restrict yourself to changes that sound easy, and do not assume any particular capability \
exists. If the honest hypothesis is that the analysis model cannot represent something the \
part is required to do, say exactly that. A hypothesis that needs software which does not \
yet exist is a VALID and valuable hypothesis -- it will be built.

2. Read the declared design intent carefully. It states what the part is required to do, \
independently of how the problem was written down. A requirement present in the intent and \
absent from the specification is a defect no stress measurement can reveal, because a load \
that was never applied leaves no trace in any field.

3. Do not propose a hypothesis whose only content is making a stated limit easier to satisfy \
unless you genuinely believe the limit itself is the defect -- and say so explicitly if you do.

4. You do not decide whether a hypothesis is right. Each one will be applied and the problem \
re-solved; the re-solve decides. You also do not decide the authority tier: propose one if you \
wish and it will be recorded, but it is assigned from the slot and the declared intent.

5. Prefer hypotheses that differ from each other in KIND, not in magnitude. Three variations \
of the same numeric change are one hypothesis, not three.

6. State which quantity your change acts on, and where you can, how the anomaly SCALES with \
it. Leverage matters: a quantity the anomaly depends on cubically is a different proposition \
from one it depends on linearly, and the largest-leverage quantity is often not the most \
sophisticated change. Do not overlook a simple change to a stated quantity because a more \
elaborate reformulation is available.

7. The declared intent lists quantities the engineer permits changing. If one of those has \
leverage on the anomaly and no earlier hypothesis has proposed changing it, that is a gap in \
the hypothesis set, not a change beneath your attention. You are told below which permitted \
quantities remain untried."""

BLIND_TOOL = {
    "name": "propose_hypotheses",
    "description": (
        "Return hypotheses about what is inadequate in the stated problem. Express each in "
        "your own terms; there is no fixed catalogue to choose from."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "hypotheses": {
                "type": "array", "minItems": 3, "maxItems": 7,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "short stable id, e.g. 'H1'"},
                        "claim": {"type": "string",
                                  "description": "what is inadequate about the stated problem, one sentence"},
                        "tuple_slot": {"type": "string",
                                       "description": "which slot of the canonical problem statement"},
                        "required_change": {
                            "type": "string",
                            "description": ("what would have to be different about the stated problem, "
                                            "concretely, in your own words -- not a function call")},
                        "quantity": {"type": "string",
                                     "description": "the specific quantity or aspect that changes"},
                        "proposed_value": {"type": "string",
                                           "description": "the value or state it should take, with units; "
                                                          "'unknown' is an acceptable answer"},
                        "needs_new_capability": {
                            "type": "boolean",
                            "description": ("true if you believe representing this requires the analysis "
                                            "model or the problem statement to gain something it does not "
                                            "currently have")},
                        "targets_anomaly": {"type": "array", "items": {"type": "string"}},
                        "proposed_tier": {"type": "string", "enum": ["T1", "T2", "T3"]},
                        "rationale": {"type": "string"},
                    },
                    "required": ["id", "claim", "tuple_slot", "required_change", "quantity",
                                 "targets_anomaly", "rationale"],
                },
            }
        },
        "required": ["hypotheses"],
    },
}


PRIOR_EVIDENCE_NOTE = """A previous round of hypotheses has already been tested by applying each \
edit and RE-SOLVING the problem. Read the outcomes below as hard evidence about the structure, \
not as feedback on your writing style:

  ELIMINATED       the edit was applied, the problem re-solved, and the anomaly still fired. That \
explanation is ruled out on its own -- do not restate it with a different number.
  INFEASIBLE       the edited specification admits no feasible design at all. Strong evidence: \
the edited requirement set is jointly unsatisfiable, so something else must give way alongside it.
  NOT_REPRESENTABLE  the analysis model has no degree of freedom to answer the question the edit \
poses. The edit is not wrong; it needs an enabling model change applied WITH it.
  NOT_ASSESSED     no trustworthy verdict; nothing follows.
  'cleared'        anomalies the edit DID remove even where the overall verdict was not survival. \
An edit that clears part of the anomaly set is a component of a coupled explanation, and the \
remaining anomalies tell you what the other components must address.

Propose a NEW set of hypotheses informed by these outcomes. Where the evidence points at a \
coupled defect spanning more than one slot, name the components that would have to be combined -- \
including any that alter what the engineer asked for. An authority stage rules on those against \
the declared intent, so do not self-censor an intent-altering component that the evidence implies. \
Pay particular attention to anomalies that NO edit has yet cleared: whatever explains them is \
still missing from the hypothesis set."""


def untried_permitted_quantities(intent, tried_quantities) -> list:
    """Quantities the intent permits changing that no hypothesis has yet proposed.

    Pure bookkeeping over the intent document and the run's own history -- it names
    no value and asserts no hypothesis. It exists because a generator reading the
    intent in prose reliably overlooks the plainest lever in it: observed on the
    L-bracket, where the intent declares the plate thickness free with stock from
    1 to 20 mm, out-of-plane stress scales as the inverse square of thickness, and
    two epochs of hypotheses proposed layer counts, aggregation functions and
    spatially varying reinforcement without once proposing a thicker plate.
    """
    if intent is None:
        return []
    tried = {str(t).lower() for t in tried_quantities}
    return [q for q in intent.mutable_geometry
            if not any(str(q).lower() in t for t in tried)]


def propose_blind(backend, spec, report, run_summary, intent=None, *,
                  max_hypotheses: int = 6, prior_evidence=None, untried=None) -> dict:
    """One model call. No operation catalogue is included in the payload."""
    import json

    payload = {
        "stated_problem": spec.to_dict(),
        "converged_run": run_summary,
        "anomaly_evidence": report.evidence,
        "fired_anomalies": [a.to_dict() for a in report.fired],
        "all_predicates_evaluated": [a.to_dict() for a in report.anomalies],
        "declared_design_intent": intent.to_dict() if intent is not None else None,
    }
    user = (
        (intent.summary_for_prompt() + "\n\n" if intent is not None else "")
        + "Converged run, anomaly evidence and the stated problem:\n"
        + json.dumps(payload, indent=2, default=str)
        + ("\n\n" + PRIOR_EVIDENCE_NOTE + "\n\nPREVIOUS ROUND OUTCOMES:\n"
           + json.dumps(prior_evidence, indent=2, default=str)
           if prior_evidence else "")
        + ("\n\nQUANTITIES THE ENGINEER PERMITS CHANGING THAT NO HYPOTHESIS HAS YET "
           "PROPOSED CHANGING: " + ", ".join(untried)
           + ". This is bookkeeping over the intent document, not a recommendation: each "
             "may or may not have leverage on this anomaly, and you decide. But do not "
             "leave one unconsidered merely because it is a simple change."
           if untried else "")
        + f"\n\nPropose at most {max_hypotheses} hypotheses that differ in kind."
    )
    raw = backend(BLIND_SYSTEM_PROMPT, user, BLIND_TOOL)
    hyps = list(raw.get("hypotheses", []))[:max_hypotheses]
    for h in hyps:
        h.setdefault("needs_new_capability", False)
        h.setdefault("proposed_value", "unknown")
    return {"model": getattr(backend, "model", "unknown"), "hypotheses": hyps,
            "usage": dict(getattr(backend, "usage", {}) or {})}

"""Deterministic authority over LLM repair proposals."""

from __future__ import annotations

from project.refinement.schema import (
    EvaluationPacket,
    PolicyDecision,
    RepairAction,
    RepairProposal,
)


# V1 deliberately supports only convergence-budget repair.
AUTO_REPAIR_TRIGGERS = {
    "design_convergence_status",
    "continuation_completion_status",
    "requested_continuation_incomplete",
}

MAX_EXTENSION_PER_ATTEMPT = 50


def decide_repair(
    proposal: RepairProposal,
    evaluation: EvaluationPacket,
    *,
    extension_preauthorized: bool,
) -> PolicyDecision:
    """Decide deterministically whether the LLM proposal may execute."""

    # ------------------------------------------------------------
    # 0. Evidence provenance
    # ------------------------------------------------------------

    allowed_evidence_keys = set(evaluation.model_dump().keys())
    unknown_keys = sorted(
        set(proposal.evidence_keys) - allowed_evidence_keys
    )

    if unknown_keys:
        return PolicyDecision(
            allowed=False,
            requires_human_approval=False,
            stop=True,
            reason=(
                "Repair proposal cited evidence fields that do not "
                f"exist: {unknown_keys}"
            ),
        )

    # ------------------------------------------------------------
    # 1. Already complete
    # ------------------------------------------------------------

    if evaluation.terminal_success:
        return PolicyDecision(
            allowed=True,
            requires_human_approval=False,
            stop=True,
            reason="Run already satisfies refinement termination criteria.",
        )

    # ------------------------------------------------------------
    # 2. Explicit stop actions are always safe
    # ------------------------------------------------------------

    if proposal.action == RepairAction.NO_ACTION:
        return PolicyDecision(
            allowed=True,
            requires_human_approval=False,
            stop=True,
            reason="Repair agent requested no further action.",
        )

    if proposal.action == RepairAction.REQUEST_HUMAN_REVIEW:
        return PolicyDecision(
            allowed=True,
            requires_human_approval=True,
            stop=True,
            reason="Repair agent requested human review.",
        )

    # ------------------------------------------------------------
    # 3. EXTEND_ITERATIONS safety gates
    # ------------------------------------------------------------

    if proposal.action == RepairAction.EXTEND_ITERATIONS:

        # Never use an execution tweak to mask an actual hard failure.
        if evaluation.hard_failed_checks:
            return PolicyDecision(
                allowed=False,
                requires_human_approval=False,
                stop=True,
                reason=(
                    "Iteration extension is forbidden while hard "
                    "deterministic validation checks are failing: "
                    f"{evaluation.hard_failed_checks}"
                ),
            )

        if not evaluation.hard_validation_passed:
            return PolicyDecision(
                allowed=False,
                requires_human_approval=False,
                stop=True,
                reason=(
                    "Iteration extension is forbidden because hard "
                    "validation did not pass."
                ),
            )

        # Do not continue if the exact source tree is not currently
        # covered by passing numerical verification.
        if not (
            evaluation.numerical_verification_current
            and evaluation.numerical_verification_passed
        ):
            return PolicyDecision(
                allowed=False,
                requires_human_approval=False,
                stop=True,
                reason=(
                    "Current source hashes are not covered by a "
                    "passing numerical verification manifest."
                ),
            )

        # Do not autonomously refine an unconfirmed interpretation
        # of the user's request.
        if evaluation.semantic_assurance_status not in {
            "fully_explicit",
            "user_confirmed",
        }:
            return PolicyDecision(
                allowed=False,
                requires_human_approval=False,
                stop=True,
                reason=(
                    "The parsed problem has not received sufficient "
                    "semantic confirmation."
                ),
            )

        triggers = (
            set(evaluation.quality_failed_checks)
            & AUTO_REPAIR_TRIGGERS
        )

        if not triggers:
            return PolicyDecision(
                allowed=False,
                requires_human_approval=False,
                stop=True,
                reason=(
                    "No convergence/continuation condition exists "
                    "that V1 is permitted to repair automatically."
                ),
            )

        requested = proposal.additional_iterations

        if requested is None:
            return PolicyDecision(
                allowed=False,
                requires_human_approval=False,
                stop=True,
                reason=(
                    "extend_iterations requires "
                    "additional_iterations."
                ),
            )

        remaining = evaluation.remaining_extra_iterations

        if remaining <= 0:
            return PolicyDecision(
                allowed=False,
                requires_human_approval=False,
                stop=True,
                reason="The authorized extra-iteration budget is exhausted.",
            )

        approved = min(
            int(requested),
            MAX_EXTENSION_PER_ATTEMPT,
            int(remaining),
        )

        if approved <= 0:
            return PolicyDecision(
                allowed=False,
                requires_human_approval=False,
                stop=True,
                reason="No positive iteration extension remains available.",
            )

        return PolicyDecision(
            allowed=True,
            requires_human_approval=not extension_preauthorized,
            stop=False,
            reason=(
                "Bounded execution-budget extension is admissible "
                f"for convergence/continuation evidence {sorted(triggers)}."
            ),
            additional_iterations=approved,
        )

    # This should be unreachable because RepairAction is an enum.
    return PolicyDecision(
        allowed=False,
        requires_human_approval=False,
        stop=True,
        reason=f"Unsupported repair action: {proposal.action}",
    )
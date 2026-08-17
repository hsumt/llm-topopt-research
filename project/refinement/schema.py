"""Typed contracts for verification-gated run refinement."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RepairAction(str, Enum):
    """Actions the LLM is allowed to propose in refinement V1."""

    NO_ACTION = "no_action"
    EXTEND_ITERATIONS = "extend_iterations"
    REQUEST_HUMAN_REVIEW = "request_human_review"


class EvaluationPacket(BaseModel):
    """Deterministic evidence supplied to the repair agent."""

    model_config = ConfigDict(extra="forbid")

    attempt: int = Field(ge=1)

    # Deterministic validation
    hard_validation_passed: bool
    hard_failed_checks: list[str] = Field(default_factory=list)
    hard_failure_reasons: list[str] = Field(default_factory=list)

    # Non-hard quality/convergence diagnostics
    quality_failed_checks: list[str] = Field(default_factory=list)
    quality_warnings: list[str] = Field(default_factory=list)

    # Optimization status
    design_converged: bool
    objective_plateau: bool
    continuation_complete: bool
    requested_continuation_complete: bool

    # Semantic/system assurance
    semantic_assurance_status: str
    numerical_verification_available: bool
    numerical_verification_current: bool
    numerical_verification_passed: bool

    # Execution budget
    original_max_iter: int = Field(ge=1)
    execution_max_iter_budget: int = Field(ge=1)
    actual_iterations: int = Field(ge=1)
    remaining_extra_iterations: int = Field(ge=0)

    # Overall refinement status
    terminal_success: bool

    # Human input is evidence, but does not override deterministic checks.
    user_feedback: str | None = None

    # Compact history only; do not recursively send whole result packets.
    previous_attempts: list[dict[str, Any]] = Field(default_factory=list)


class RepairProposal(BaseModel):
    """One bounded action proposed by the LLM."""

    model_config = ConfigDict(extra="forbid")

    action: RepairAction

    rationale: str = Field(min_length=1)

    # These must name fields from EvaluationPacket.
    evidence_keys: list[str] = Field(min_length=1)

    additional_iterations: int | None = Field(
        default=None,
        ge=1,
        le=100,
    )

    feedback_response: str | None = None


class PolicyDecision(BaseModel):
    """Deterministic decision about whether a proposal may execute."""

    model_config = ConfigDict(extra="forbid")

    allowed: bool
    requires_human_approval: bool
    stop: bool

    reason: str

    additional_iterations: int = Field(default=0, ge=0)
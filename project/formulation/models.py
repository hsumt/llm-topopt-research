"""Shared data models for the pre-solve formulation conversation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from project.parser.schema import ParserResult, ProblemRoute


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ConversationMessage(StrictModel):
    role: Literal["user", "assistant", "system"]
    content: str


class AttachmentRef(StrictModel):
    """Serializable metadata for a user-supplied context asset."""

    id: str
    name: str
    media_type: str
    size_bytes: int
    kind: Literal["image", "pdf", "text", "other"]


class ContextSnippet(StrictModel):
    """A deterministic retrieval result from user-supplied context."""

    id: str
    source: str
    text: str
    score: float = 0.0


class FormulationConcern(StrictModel):
    id: str
    category: Literal[
        "physics_model",
        "geometry",
        "material",
        "boundary_condition",
        "load_or_source",
        "initial_condition",
        "objective",
        "constraint",
        "design_variable",
        "coupling",
        "manufacturing",
        "units",
        "context",
        "contradiction",
        "other",
    ]
    severity: Literal["low", "medium", "high"]
    blocking: bool
    description: str
    rationale: str
    related_fields: list[str] = Field(default_factory=list)
    related_context_ids: list[str] = Field(default_factory=list)


class ClarificationQuestion(StrictModel):
    """One user decision inside a batch clarification packet."""

    id: str
    prompt: str
    why_needed: str
    issue_ids: list[str] = Field(default_factory=list)
    category: Literal[
        "physics_model",
        "geometry",
        "material",
        "boundary_condition",
        "load_or_source",
        "initial_condition",
        "objective",
        "constraint",
        "design_variable",
        "coupling",
        "manufacturing",
        "units",
        "context",
        "contradiction",
        "other",
    ] = "other"
    answer_type: Literal[
        "text",
        "single_choice",
        "multiple_choice",
        "visual_reference",
    ] = "text"
    options: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    related_context_ids: list[str] = Field(default_factory=list)


class ClarificationPacket(StrictModel):
    """Questions that can be answered together without dependency conflicts."""

    questions: list[ClarificationQuestion] = Field(default_factory=list)


class CriticResult(StrictModel):
    status: Literal[
        "needs_clarification",
        "ready_for_review",
        "blocked_by_contradiction",
    ]
    summary: str
    concerns: list[FormulationConcern] = Field(default_factory=list)
    clarification_packet: ClarificationPacket = Field(
        default_factory=ClarificationPacket
    )
    relevant_context_ids: list[str] = Field(default_factory=list)


class PatchOperation(StrictModel):
    op: Literal["add", "replace", "remove"]
    path: str
    value: object | None = None


class ResolutionProposal(StrictModel):
    action: Literal["apply", "no_change", "cancel"]
    operations: list[PatchOperation] = Field(default_factory=list)
    resolved_unresolved_ids: list[str] = Field(default_factory=list)
    resolved_contradiction_ids: list[str] = Field(default_factory=list)
    incorporated_context_ids: list[str] = Field(default_factory=list)
    assistant_note: str = ""


class RevisionRecord(StrictModel):
    revision: int
    user_message: str
    structured_answers: dict[str, str] = Field(default_factory=dict)
    operations: list[PatchOperation] = Field(default_factory=list)
    resolved_unresolved_ids: list[str] = Field(default_factory=list)
    resolved_contradiction_ids: list[str] = Field(default_factory=list)
    incorporated_context_ids: list[str] = Field(default_factory=list)


class ReadinessReport(StrictModel):
    ready: bool
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class FormulationSession(StrictModel):
    original_problem: str
    supplied_context: str | None = None
    attachments: list[AttachmentRef] = Field(default_factory=list)
    retrieved_context: list[ContextSnippet] = Field(default_factory=list)
    route: ProblemRoute
    parser_result: ParserResult
    critic_result: CriticResult
    messages: list[ConversationMessage] = Field(default_factory=list)
    revisions: list[RevisionRecord] = Field(default_factory=list)
    usage: list[dict] = Field(default_factory=list)
    revision: int = 0
    status: Literal[
        "reviewing",
        "awaiting_user",
        "ready_for_approval",
        "approved",
        "cancelled",
    ] = "reviewing"

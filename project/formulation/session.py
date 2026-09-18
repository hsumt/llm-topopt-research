"""Orchestration for the complete pre-solve formulation conversation."""

from __future__ import annotations

from typing import Callable

from project.formulation.context import (
    ContextAttachment,
    attachment_ref,
    build_context_snippets,
    retrieval_query_from_state,
    retrieve_context,
)
from project.formulation.dialogue import compose_critic_message
from project.formulation.models import (
    ConversationMessage,
    FormulationSession,
    RevisionRecord,
)
from project.formulation.patching import apply_resolution
from project.formulation.verification import check_readiness
from project.llm.formulation_critic import review_formulation
from project.llm.formulation_resolver import resolve_user_reply
from project.parser.client import parse_problem
from project.parser.router import route_problem
from project.parser.schema import ProblemRoute


_CANCEL_PHRASES = {"cancel", "stop", "quit", "end session"}
_APPROVE_PHRASES = {"approve", "approved", "looks good", "accept formulation"}

ProgressCallback = Callable[[str], None]


def _report(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _sync_route_with_spec(route: ProblemRoute, session: FormulationSession) -> ProblemRoute:
    spec = session.parser_result.spec
    families = []
    for physics in spec.physics:
        if physics.family not in families:
            families.append(physics.family)

    dimension = route.spatial_dimension
    if spec.geometry and spec.geometry.spatial_dimension is not None:
        dimension = spec.geometry.spatial_dimension

    return route.model_copy(
        update={
            "task_type": spec.problem_kind,
            "physics_families": families or route.physics_families,
            "spatial_dimension": dimension,
            "multiphysics": bool(len(families) > 1 or spec.couplings),
        }
    )


def _set_status_from_current_state(session: FormulationSession) -> None:
    readiness = check_readiness(session)
    if readiness.ready and session.critic_result.status == "ready_for_review":
        session.status = "ready_for_approval"
    else:
        session.status = "awaiting_user"


def _format_structured_answers(answers: dict[str, str]) -> str:
    nonblank = [(key, value.strip()) for key, value in answers.items() if value.strip()]
    if not nonblank:
        return ""
    lines = ["Clarification packet answers:"]
    lines.extend(f"- {key}: {value}" for key, value in nonblank)
    return "\n".join(lines)


def _retrieve_for_parser(
    problem: str,
    context: str | None,
    attachments: list[ContextAttachment] | None,
):
    all_snippets = build_context_snippets(context, attachments)
    return retrieve_context(problem, all_snippets, limit=6)


def _retrieve_for_review(
    problem: str,
    context: str | None,
    parser_result,
    attachments: list[ContextAttachment] | None,
):
    all_snippets = build_context_snippets(context, attachments)
    query = retrieval_query_from_state(problem, parser_result)
    return retrieve_context(query, all_snippets, limit=6)


def start_session(
    problem: str,
    context: str | None = None,
    *,
    attachments: list[ContextAttachment] | None = None,
    progress: ProgressCallback | None = None,
) -> FormulationSession:
    """Create a new formulation session from one engineering request."""

    attachments = attachments or []
    _report(progress, "Classifying the engineering task and physics family...")
    route, route_usage = route_problem(problem, context)
    _report(progress, "Searching supplied context and uploaded documents for relevant evidence...")
    initial_context = _retrieve_for_parser(problem, context, attachments)
    _report(progress, f"Found {len(initial_context)} relevant context snippet(s) for the initial parse.")
    parser_result, parser_usage = parse_problem(
        problem,
        route=route,
        context=context,
        retrieved_context=initial_context,
        attachments=attachments,
        progress=progress,
    )
    _report(progress, "Searching context again using the unresolved formulation issues...")
    _report(progress, "Re-searching context using the updated formulation...")
    review_context = _retrieve_for_review(
        problem,
        context,
        parser_result,
        attachments,
    )

    _report(progress, "Reviewing the draft formulation for missing, ambiguous, or contradictory engineering intent...")
    critic, critic_usage = review_formulation(
        problem=problem,
        context=context,
        route=route,
        parser_result=parser_result,
        retrieved_context=[item.model_dump() for item in review_context],
        attachments=attachments,
    )

    _report(progress, "Building the clarification packet and engineer-facing formulation mirrors...")
    session = FormulationSession(
        original_problem=problem,
        supplied_context=context.strip() if context and context.strip() else None,
        attachments=[attachment_ref(item) for item in attachments],
        retrieved_context=review_context,
        route=route,
        parser_result=parser_result,
        critic_result=critic,
        messages=[
            ConversationMessage(role="user", content=problem.strip()),
            ConversationMessage(
                role="assistant",
                content=compose_critic_message(critic, parser_result),
            ),
        ],
        usage=[route_usage, parser_usage, critic_usage],
        status="reviewing",
    )
    _set_status_from_current_state(session)
    _report(progress, "Initial formulation review complete.")
    return session


def approve_session(session: FormulationSession) -> FormulationSession:
    updated = session.model_copy(deep=True)
    readiness = check_readiness(updated)
    if not readiness.ready or updated.critic_result.status != "ready_for_review":
        details = "; ".join(readiness.blockers) or "critic still requires clarification"
        raise ValueError(f"Formulation cannot be approved yet: {details}")

    updated.status = "approved"
    updated.messages.append(
        ConversationMessage(
            role="assistant",
            content=(
                "Formulation approved. The pre-solve workflow stops here; no solver "
                "has been selected or executed."
            ),
        )
    )
    return updated


def continue_session(
    session: FormulationSession,
    user_message: str = "",
    *,
    structured_answers: dict[str, str] | None = None,
    attachments: list[ContextAttachment] | None = None,
    progress: ProgressCallback | None = None,
) -> FormulationSession:
    """Apply one free-form clarification or one batch of structured answers."""

    if session.status in {"approved", "cancelled"}:
        raise ValueError(f"Session is already {session.status}.")

    structured_answers = structured_answers or {}
    packet_text = _format_structured_answers(structured_answers)
    display_message = packet_text or user_message.strip()
    if not display_message:
        raise ValueError("A user clarification or at least one packet answer is required")

    normalized = display_message.strip().lower()
    if session.status == "ready_for_approval" and normalized in _APPROVE_PHRASES:
        return approve_session(session)

    updated = session.model_copy(deep=True)
    updated.messages.append(ConversationMessage(role="user", content=display_message))

    if normalized in _CANCEL_PHRASES:
        updated.status = "cancelled"
        updated.messages.append(
            ConversationMessage(role="assistant", content="Formulation session cancelled.")
        )
        return updated

    _report(progress, "Interpreting the clarification packet as a structured specification update...")
    proposal, resolver_usage = resolve_user_reply(
        updated,
        display_message,
        structured_answers=structured_answers,
        attachments=attachments,
    )
    updated.usage.append(resolver_usage)

    if proposal.action == "cancel":
        updated.status = "cancelled"
        updated.messages.append(
            ConversationMessage(role="assistant", content="Formulation session cancelled.")
        )
        return updated

    if proposal.action == "apply":
        _report(progress, "Applying the proposed patch deterministically and re-validating the ProblemSpec...")
        try:
            updated.parser_result = apply_resolution(
                updated.parser_result,
                proposal,
                user_message=display_message,
            )
        except Exception as exc:
            updated.messages.append(
                ConversationMessage(
                    role="assistant",
                    content=(
                        "I could not safely apply that clarification to the structured "
                        f"problem definition ({exc}). Please restate the intended change."
                    ),
                )
            )
            updated.status = "awaiting_user"
            return updated

        updated.revision += 1
        updated.revisions.append(
            RevisionRecord(
                revision=updated.revision,
                user_message=display_message,
                structured_answers={
                    key: value for key, value in structured_answers.items() if value.strip()
                },
                operations=proposal.operations,
                resolved_unresolved_ids=proposal.resolved_unresolved_ids,
                resolved_contradiction_ids=proposal.resolved_contradiction_ids,
                incorporated_context_ids=proposal.incorporated_context_ids,
            )
        )
        updated.route = _sync_route_with_spec(updated.route, updated)

    review_context = _retrieve_for_review(
        updated.original_problem,
        updated.supplied_context,
        updated.parser_result,
        attachments,
    )
    updated.retrieved_context = review_context

    _report(progress, "Re-reviewing the updated formulation and preparing the next packet...")
    critic, critic_usage = review_formulation(
        problem=updated.original_problem,
        context=updated.supplied_context,
        route=updated.route,
        parser_result=updated.parser_result,
        retrieved_context=[item.model_dump() for item in review_context],
        attachments=attachments,
        recent_user_clarification=display_message,
    )
    updated.critic_result = critic
    updated.usage.append(critic_usage)

    message = compose_critic_message(critic, updated.parser_result)
    if proposal.assistant_note.strip() and proposal.action == "no_change":
        message = f"{proposal.assistant_note.strip()}\n\n{message}"
    updated.messages.append(ConversationMessage(role="assistant", content=message))

    _set_status_from_current_state(updated)
    _report(progress, "Clarification round complete.")
    return updated

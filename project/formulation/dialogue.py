"""Presentation-neutral dialogue helpers for formulation sessions."""

from __future__ import annotations

from project.formulation.models import CriticResult
from project.parser.schema import ParserResult


def compose_critic_message(
    critic: CriticResult,
    parser_result: ParserResult,
) -> str:
    parts: list[str] = []

    if critic.summary.strip():
        parts.append(critic.summary.strip())

    if critic.relevant_context_ids:
        by_id = {item.id: item for item in parser_result.context_candidates}
        selected = [
            by_id[item_id]
            for item_id in critic.relevant_context_ids
            if item_id in by_id
        ]
        if selected:
            lines = ["Relevant supplied context:"]
            for item in selected:
                marker = (
                    "incorporated"
                    if item.incorporated_into_spec
                    else "not yet incorporated"
                )
                lines.append(
                    f"- {item.text} ({item.relevance}; {marker})"
                )
            parts.append("\n".join(lines))

    if critic.status == "ready_for_review":
        parts.append(
            "The current formulation has no blocking pre-solve issues. Review the "
            "problem/context mirrors and structured specification before approval."
        )
    else:
        packet = critic.clarification_packet.questions
        if packet:
            lines = [
                f"I have {len(packet)} clarification item(s) that can be answered together:"
            ]
            for index, question in enumerate(packet, start=1):
                lines.append(f"{index}. {question.prompt}")
            parts.append("\n".join(lines))

    return "\n\n".join(part for part in parts if part)

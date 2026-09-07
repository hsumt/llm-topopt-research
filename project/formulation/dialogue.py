"""Interactive formulation-review loop.

This is the conversational layer between the parser and the deterministic
solver. The LLM critic proposes questions; the human answers; the resolver
interprets the answer; deterministic Python authorizes and applies only the
fields attached to that question; then the critic re-runs on the revised spec.
"""

from __future__ import annotations

from pathlib import Path

from project.formulation.patching import (
    apply_resolution,
    provenance_models,
    synchronize_final_values,
)
from project.formulation.preview import (
    format_intent_snapshot,
    generate_intent_preview,
)
from project.formulation.session import (
    append_event,
    create_session,
    finalize_session,
    initialize_session,
)
from project.llm.formulation_critic import (
    FormulationIssue,
    _policy_accept_issue,
    _provenance_index,
    critique_formulation,
)
from project.llm.formulation_resolver import resolve_formulation_answer
from project.parser.provenance import summarize_semantic_assurance

CANCEL_PHRASES = {"cancel", "quit", "stop", "abort"}
USE_AS_IS_PHRASES = {"use as is", "run as is", "accept as is", "proceed as is"}
KEEP_CURRENT_PHRASES = {
    "keep current",
    "keep it",
    "current is fine",
    "that's fine",
    "thats fine",
    "yes current",
}
YES_PHRASES = {"y", "yes", "confirm", "correct", "looks right", "looks good"}


def _zero_usage() -> dict:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _add_usage(total: dict, usage: dict) -> None:
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        total[key] += int(usage.get(key, 0))


def _print_issue(issue: dict, print_fn=print) -> None:
    fields = ", ".join(issue.get("affected_fields", [])) or "cross-cutting"
    print_fn(
        f"[{issue['severity'].upper()}] {issue['category']} ({fields})"
    )
    print_fn(f"  Observation: {issue['observation']}")
    print_fn(f"  Why it matters: {issue['why_it_matters']}")
    print_fn(f"  Question: {issue['question_for_user']}")


def _issue_signature(issue: dict) -> tuple[str, tuple[str, ...]]:
    return (
        str(issue.get("category", "other")),
        tuple(sorted(str(path) for path in issue.get("affected_fields", []))),
    )


def _merge_active_issues(
    *,
    new_issues: list[dict],
    pending_issues: dict[tuple[str, tuple[str, ...]], dict],
    final_field_provenance: list[dict],
) -> tuple[list[dict], dict[tuple[str, tuple[str, ...]], dict], int]:
    """Retain previously surfaced unresolved concerns across LLM re-critiques.

    A later LLM call is allowed to discover new concerns, but it cannot silently
    make an earlier independent concern disappear. Pending concerns are removed
    only when the deterministic provenance policy says their fields have been
    explicitly settled or they otherwise leave the critic's allowed scope.
    """
    provenance = _provenance_index(final_field_provenance)
    merged = dict(pending_issues)
    new_signatures = set()

    for issue in new_issues:
        signature = _issue_signature(issue)
        merged[signature] = dict(issue)
        new_signatures.add(signature)

    active: list[dict] = []
    retained: dict[tuple[str, tuple[str, ...]], dict] = {}
    carried_forward = 0
    for signature, issue in merged.items():
        try:
            model = FormulationIssue.model_validate(issue)
            keep, _ = _policy_accept_issue(model, provenance)
        except Exception:
            keep = False
        if not keep:
            continue
        retained[signature] = issue
        active.append(issue)
        if signature not in new_signatures:
            carried_forward += 1

    severity_rank = {"high": 0, "medium": 1, "low": 2}
    active.sort(key=lambda item: severity_rank.get(item.get("severity"), 3))
    return active, retained, carried_forward


def _write_live_preview(
    *,
    spec,
    session_dir: Path,
    critic_round: int,
    stage: str,
    focus: str,
    print_fn=print,
) -> Path | None:
    """Save and announce a deterministic preview for the current dialogue state."""
    safe_focus = "".join(ch if ch.isalnum() else "_" for ch in focus).strip("_")
    safe_focus = safe_focus[:36] or "review"
    path = session_dir / f"round_{critic_round:02d}_{stage}_{safe_focus}.png"
    try:
        result = generate_intent_preview(
            spec,
            path,
            title=f"Current intent — {focus.replace('_', ' ')}",
        )
    except Exception as exc:
        print_fn(f"  Live intent preview unavailable: {exc}")
        return None

    print_fn("\nCurrent structured interpretation:")
    print_fn(format_intent_snapshot(spec))
    print_fn(f"  Live preview PNG: {result}")
    return result


def _finalize_pending_interactions(records: list[dict]) -> list[dict]:
    finalized = []
    for record in records:
        record = dict(record)
        if record.get("interaction_status") in {
            "pending_formulation_review",
            "pending_final_preview_confirmation",
        }:
            record["interaction_status"] = "confirmed_in_final_intent_preview"
        finalized.append(record)
    return finalized


def run_formulation_dialogue(
    *,
    original_prompt: str,
    spec,
    final_field_provenance: list[dict],
    input_fn=input,
    print_fn=print,
    max_critic_rounds: int = 5,
    max_answer_turns: int = 8,
) -> dict:
    """Run critic -> question -> answer -> restricted patch -> re-critic loop."""

    session_id, session_dir = create_session(original_prompt)
    current_spec = spec
    current_provenance = synchronize_final_values(
        current_spec, final_field_provenance
    )
    initialize_session(
        session_id=session_id,
        session_dir=session_dir,
        original_prompt=original_prompt,
        initial_spec=current_spec.model_dump(),
        initial_field_provenance=current_provenance,
    )

    dialogue_history: list[dict] = []
    critic_usage_total = _zero_usage()
    resolver_usage_total = _zero_usage()
    critic_calls: list[dict] = []
    resolution_audits: list[dict] = []
    answer_turns = 0
    accepted_as_is = False
    last_critique: dict = {
        "status": "not_run",
        "review_recommended": None,
        "summary": "Formulation critic has not run.",
        "issues": [],
    }
    status = "in_progress"
    pending_issues: dict[tuple[str, tuple[str, ...]], dict] = {}

    print_fn("\n=== Interactive Problem-Formulation Review ===")
    print_fn(
        "The critic may ask about consequential interpretation choices. "
        "It cannot silently edit the problem."
    )
    print_fn(
        "Commands: 'keep current' confirms the current interpretation; "
        "'use as is' ends questioning; 'cancel' stops before FEA."
    )

    for critic_round in range(1, max_critic_rounds + 1):
        critique, usage = critique_formulation(
            original_prompt=original_prompt,
            spec=current_spec,
            final_field_provenance=current_provenance,
            dialogue_history=dialogue_history,
        )
        _add_usage(critic_usage_total, usage)
        last_critique = critique
        critic_calls.append(
            {
                "round": critic_round,
                "spec": current_spec.model_dump(),
                "field_provenance": current_provenance,
                "result": critique,
                "usage": usage,
            }
        )
        append_event(
            session_dir,
            {
                "type": "critic_result",
                "round": critic_round,
                "result": critique,
                "usage": usage,
                "spec": current_spec.model_dump(),
            },
        )

        print_fn("\n--- Formulation Critic ---")
        print_fn(critique.get("summary", ""))

        if critique.get("status") == "unavailable":
            print_fn("The formulation critic did not complete successfully.")
            answer = input_fn(
                "Continue to the deterministic intent preview without this review? [y/N]\n> "
            ).strip().lower()
            if answer not in YES_PHRASES:
                status = "cancelled_critic_unavailable"
                finalize_session(
                    session_id=session_id,
                    session_dir=session_dir,
                    status=status,
                    final_spec=current_spec.model_dump(),
                    final_field_provenance=current_provenance,
                    summary={
                        "critic_calls": len(critic_calls),
                        "resolution_count": len(resolution_audits),
                        "critic_usage": critic_usage_total,
                        "resolver_usage": resolver_usage_total,
                        "human_final_confirmation": False,
                    },
                )
                return {
                    "cancelled": True,
                    "session_id": session_id,
                    "session_dir": str(session_dir),
                }
            accepted_as_is = True
            break

        issues, pending_issues, carried_forward = _merge_active_issues(
            new_issues=list(critique.get("issues", [])),
            pending_issues=pending_issues,
            final_field_provenance=current_provenance,
        )
        if carried_forward:
            print_fn(
                f"Retaining {carried_forward} previously identified unresolved "
                "concern(s) that remain admissible after the latest update."
            )
        if not issues:
            print_fn("No unresolved grounded formulation question was identified.")
            break

        # Process one question at a time, then re-criticize the revised spec.
        issue = dict(issues[0])
        issue["issue_id"] = f"r{critic_round}_i1"
        live_before = _write_live_preview(
            spec=current_spec,
            session_dir=session_dir,
            critic_round=critic_round,
            stage="before",
            focus=issue.get("category", "review"),
            print_fn=print_fn,
        )
        if live_before is not None:
            append_event(
                session_dir,
                {
                    "type": "intent_preview",
                    "round": critic_round,
                    "stage": "before_question",
                    "issue_category": issue.get("category"),
                    "preview_path": str(live_before),
                    "spec": current_spec.model_dump(),
                },
            )
        _print_issue(issue, print_fn=print_fn)

        if not issue.get("affected_fields"):
            print_fn(
                "  This concern is not editable inside the current ProblemSpec "
                "schema. It may represent unsupported or missing problem content."
            )
            answer = input_fn(
                "Type 'use as is' to acknowledge it, or 'cancel' to reformulate.\n> "
            ).strip().lower()
            if answer in USE_AS_IS_PHRASES:
                accepted_as_is = True
                dialogue_history.append(
                    {
                        "role": "user",
                        "action": "accepted_cross_cutting_issue_as_is",
                        "issue": issue,
                    }
                )
                append_event(
                    session_dir,
                    {
                        "type": "human_accept_as_is",
                        "issue": issue,
                        "scope": "cross_cutting",
                    },
                )
                break
            status = "cancelled_cross_cutting_issue"
            finalize_session(
                session_id=session_id,
                session_dir=session_dir,
                status=status,
                final_spec=current_spec.model_dump(),
                final_field_provenance=current_provenance,
                summary={
                    "critic_calls": len(critic_calls),
                    "resolution_count": len(resolution_audits),
                    "critic_usage": critic_usage_total,
                    "resolver_usage": resolver_usage_total,
                    "human_final_confirmation": False,
                },
            )
            return {
                "cancelled": True,
                "session_id": session_id,
                "session_dir": str(session_dir),
            }

        follow_up = issue["question_for_user"]
        resolved_this_round = False

        while answer_turns < max_answer_turns and not resolved_this_round:
            answer_turns += 1
            user_answer = input_fn(f"\n{follow_up}\n> ").strip()
            lowered = user_answer.lower()

            if lowered in CANCEL_PHRASES:
                status = "cancelled_by_user"
                finalize_session(
                    session_id=session_id,
                    session_dir=session_dir,
                    status=status,
                    final_spec=current_spec.model_dump(),
                    final_field_provenance=current_provenance,
                    summary={
                        "critic_calls": len(critic_calls),
                        "resolution_count": len(resolution_audits),
                        "critic_usage": critic_usage_total,
                        "resolver_usage": resolver_usage_total,
                        "human_final_confirmation": False,
                    },
                )
                return {
                    "cancelled": True,
                    "session_id": session_id,
                    "session_dir": str(session_dir),
                }

            if lowered in USE_AS_IS_PHRASES:
                accepted_as_is = True
                dialogue_history.append(
                    {
                        "role": "user",
                        "action": "accepted_current_spec_as_is",
                        "answer": user_answer,
                        "issue": issue,
                    }
                )
                append_event(
                    session_dir,
                    {
                        "type": "human_accept_as_is",
                        "issue": issue,
                        "user_answer": user_answer,
                        "scope": "all_remaining",
                    },
                )
                resolved_this_round = True
                break

            if lowered in KEEP_CURRENT_PHRASES:
                resolution = {
                    "status": "confirm_current",
                    "updates": [],
                    "confirmed_fields": list(issue["affected_fields"]),
                    "interpretation": "User explicitly confirmed the current values.",
                    "follow_up_question": None,
                }
                resolver_usage = _zero_usage()
            else:
                try:
                    resolution, resolver_usage = resolve_formulation_answer(
                        spec=current_spec,
                        issue=issue,
                        user_answer=user_answer,
                        dialogue_history=dialogue_history,
                    )
                    _add_usage(resolver_usage_total, resolver_usage)
                except Exception as exc:
                    print_fn(
                        f"Could not interpret that clarification safely ({exc}). "
                        "Please answer again or type 'cancel'."
                    )
                    append_event(
                        session_dir,
                        {
                            "type": "resolver_error",
                            "issue": issue,
                            "user_answer": user_answer,
                            "error": str(exc),
                        },
                    )
                    continue

            append_event(
                session_dir,
                {
                    "type": "resolver_result",
                    "issue": issue,
                    "user_answer": user_answer,
                    "resolution": resolution,
                    "usage": resolver_usage,
                },
            )

            if resolution["status"] == "ask_again":
                follow_up = resolution["follow_up_question"]
                print_fn(
                    "The answer still leaves a field-level ambiguity, so no "
                    "specification change was applied."
                )
                dialogue_history.extend(
                    [
                        {"role": "user", "answer": user_answer, "issue": issue},
                        {
                            "role": "resolver",
                            "status": "ask_again",
                            "question": follow_up,
                        },
                    ]
                )
                continue

            try:
                new_spec, new_provenance, audit = apply_resolution(
                    spec=current_spec,
                    final_field_provenance=current_provenance,
                    issue=issue,
                    resolution=resolution,
                    user_answer=user_answer,
                )
            except Exception as exc:
                print_fn(
                    f"Proposed clarification was rejected by the deterministic "
                    f"patch gate ({exc}). Please answer again or type 'cancel'."
                )
                append_event(
                    session_dir,
                    {
                        "type": "patch_rejected",
                        "issue": issue,
                        "user_answer": user_answer,
                        "resolution": resolution,
                        "error": str(exc),
                    },
                )
                continue

            current_spec = new_spec
            current_provenance = new_provenance
            resolution_audits.append(audit)
            dialogue_history.extend(
                [
                    {"role": "user", "answer": user_answer, "issue": issue},
                    {"role": "system", "applied_resolution": audit},
                ]
            )
            append_event(
                session_dir,
                {
                    "type": "patch_applied",
                    "issue": issue,
                    "audit": audit,
                    "spec_after": current_spec.model_dump(),
                },
            )

            if audit["updates"]:
                live_after = _write_live_preview(
                    spec=current_spec,
                    session_dir=session_dir,
                    critic_round=critic_round,
                    stage="after",
                    focus=issue.get("category", "review"),
                    print_fn=print_fn,
                )
                if live_after is not None:
                    append_event(
                        session_dir,
                        {
                            "type": "intent_preview",
                            "round": critic_round,
                            "stage": "after_update",
                            "issue_category": issue.get("category"),
                            "preview_path": str(live_after),
                            "spec": current_spec.model_dump(),
                        },
                    )
                print_fn("\nApplied user-authorized formulation update:")
                for update in audit["updates"]:
                    print_fn(
                        f"  {update['field_path']}: "
                        f"{update['previous_value']!r} -> {update['new_value']!r}"
                    )
            else:
                print_fn(
                    "\nCurrent interpretation explicitly confirmed for: "
                    + ", ".join(audit["confirmed_fields"])
                )
            resolved_this_round = True

        if accepted_as_is:
            break

        if not resolved_this_round:
            print_fn(
                "Maximum clarification turns reached before the issue was resolved."
            )
            status = "cancelled_turn_limit"
            finalize_session(
                session_id=session_id,
                session_dir=session_dir,
                status=status,
                final_spec=current_spec.model_dump(),
                final_field_provenance=current_provenance,
                summary={
                    "critic_calls": len(critic_calls),
                    "resolution_count": len(resolution_audits),
                    "critic_usage": critic_usage_total,
                    "resolver_usage": resolver_usage_total,
                    "human_final_confirmation": False,
                },
            )
            return {
                "cancelled": True,
                "session_id": session_id,
                "session_dir": str(session_dir),
            }
    else:
        # Exhausted critic rounds with at least one issue still present.
        print_fn("\nMaximum formulation-critic rounds reached.")
        answer = input_fn("Proceed with the current spec anyway? [y/N]\n> ").strip().lower()
        if answer not in YES_PHRASES:
            status = "cancelled_round_limit"
            finalize_session(
                session_id=session_id,
                session_dir=session_dir,
                status=status,
                final_spec=current_spec.model_dump(),
                final_field_provenance=current_provenance,
                summary={
                    "critic_calls": len(critic_calls),
                    "resolution_count": len(resolution_audits),
                    "critic_usage": critic_usage_total,
                    "resolver_usage": resolver_usage_total,
                    "human_final_confirmation": False,
                },
            )
            return {
                "cancelled": True,
                "session_id": session_id,
                "session_dir": str(session_dir),
            }
        accepted_as_is = True

    print_fn("\n--- Final formulation after dialogue ---")
    print_fn(current_spec.model_dump_json(indent=2))

    preview_path: Path | None = None
    try:
        preview_path = generate_intent_preview(
            current_spec, session_dir / "intent_preview.png"
        )
        print_fn(f"\nDeterministic intent preview saved -> {preview_path}")
        print_fn(
            "Open that PNG if useful. It is generated directly from the exact "
            "ProblemSpec shown above."
        )
    except Exception as exc:
        print_fn(f"\nIntent preview generation failed: {exc}")
        answer = input_fn(
            "Continue without the required diagram confirmation? [y/N]\n> "
        ).strip().lower()
        if answer not in YES_PHRASES:
            status = "cancelled_preview_failure"
            finalize_session(
                session_id=session_id,
                session_dir=session_dir,
                status=status,
                final_spec=current_spec.model_dump(),
                final_field_provenance=current_provenance,
                summary={
                    "critic_calls": len(critic_calls),
                    "resolution_count": len(resolution_audits),
                    "critic_usage": critic_usage_total,
                    "resolver_usage": resolver_usage_total,
                    "human_final_confirmation": False,
                },
            )
            return {
                "cancelled": True,
                "session_id": session_id,
                "session_dir": str(session_dir),
            }

    confirmation = input_fn(
        "\nDoes this final specification/intent preview match the problem you want solved? [y/N]\n> "
    ).strip().lower()
    final_preview_confirmed = confirmation in YES_PHRASES
    append_event(
        session_dir,
        {
            "type": "final_intent_confirmation",
            "confirmed": final_preview_confirmed,
            "spec": current_spec.model_dump(),
            "preview_path": str(preview_path) if preview_path else None,
        },
    )

    if not final_preview_confirmed:
        status = "rejected_final_intent_preview"
        print_fn(
            "Run cancelled before deterministic optimization. The complete "
            "formulation session has been saved so the mismatch is auditable."
        )
        finalize_session(
            session_id=session_id,
            session_dir=session_dir,
            status=status,
            final_spec=current_spec.model_dump(),
            final_field_provenance=current_provenance,
            summary={
                "critic_calls": len(critic_calls),
                "resolution_count": len(resolution_audits),
                "critic_usage": critic_usage_total,
                "resolver_usage": resolver_usage_total,
                "human_final_confirmation": False,
            },
            preview_path=preview_path,
        )
        return {
            "cancelled": True,
            "session_id": session_id,
            "session_dir": str(session_dir),
        }

    current_provenance = _finalize_pending_interactions(current_provenance)
    semantic_assurance = summarize_semantic_assurance(
        provenance_models(current_provenance),
        final_preview_confirmed=True,
    )

    formulation_total_usage = {
        "critic": critic_usage_total,
        "resolver": resolver_usage_total,
        "input_tokens": critic_usage_total["input_tokens"]
        + resolver_usage_total["input_tokens"],
        "output_tokens": critic_usage_total["output_tokens"]
        + resolver_usage_total["output_tokens"],
        "total_tokens": critic_usage_total["total_tokens"]
        + resolver_usage_total["total_tokens"],
    }

    status = "confirmed_for_execution"
    finalize_session(
        session_id=session_id,
        session_dir=session_dir,
        status=status,
        final_spec=current_spec.model_dump(),
        final_field_provenance=current_provenance,
        summary={
            "critic_calls": len(critic_calls),
            "resolution_count": len(resolution_audits),
            "accepted_as_is": accepted_as_is,
            "critic_usage": critic_usage_total,
            "resolver_usage": resolver_usage_total,
            "formulation_agent_usage": formulation_total_usage,
            "human_final_confirmation": True,
            "semantic_assurance": semantic_assurance,
        },
        preview_path=preview_path,
    )

    return {
        "cancelled": False,
        "spec": current_spec,
        "final_field_provenance": current_provenance,
        "semantic_assurance": semantic_assurance,
        "session_id": session_id,
        "session_dir": str(session_dir),
        "preview_path": str(preview_path) if preview_path else None,
        "critic_calls": critic_calls,
        "resolution_audits": resolution_audits,
        "accepted_as_is": accepted_as_is,
        "last_critique": last_critique,
        "usage": formulation_total_usage,
    }

"""Outer run -> evaluate -> propose -> gate -> rerun controller."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable

from project.llm.critic import criticize
from project.llm.repair_agent import propose_repair
from project.paths import RUNS_ROOT
from project.refinement.evaluator import build_evaluation_packet
from project.refinement.policy import decide_repair
from project.refinement.schema import (
    EvaluationPacket,
    RepairAction,
    RepairProposal,
    PolicyDecision,
)
from project.topopt.controller import main_from_spec


FeedbackCallback = Callable[[EvaluationPacket], str | None]
ApprovalCallback = Callable[
    [RepairProposal, PolicyDecision, EvaluationPacket],
    bool,
]


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")

    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _compact_attempt(record: dict) -> dict:
    """Return only small evidence needed by the next repair-agent call."""

    evaluation = record["evaluation"]
    proposal = record.get("proposal")
    policy = record.get("policy")

    return {
        "attempt": record["attempt"],
        "execution_max_iter_budget": evaluation[
            "execution_max_iter_budget"
        ],
        "actual_iterations": evaluation[
            "actual_iterations"
        ],
        "hard_validation_passed": evaluation[
            "hard_validation_passed"
        ],
        "hard_failed_checks": evaluation["hard_failed_checks"],
        "quality_failed_checks": evaluation[
            "quality_failed_checks"
        ],
        "design_converged": evaluation["design_converged"],
        "requested_continuation_complete": evaluation[
            "requested_continuation_complete"
        ],
        "proposal": proposal,
        "policy": policy,
    }


def _attach_final_critic(
    packet: dict,
    *,
    attempt_dir: Path,
    history: list[dict],
) -> tuple[dict, dict]:
    """Run the ordinary evidence-limited critic once, after final success."""

    critic_input = dict(packet["what_claude_saw"])

    critic_input["refinement_context"] = {
        "attempt_count": len(history),
        "attempts": [_compact_attempt(item) for item in history],
    }

    summary, usage = criticize(critic_input)

    packet["critic_agent_summary"] = summary
    packet["what_claude_saw"] = critic_input

    packet.setdefault("compute_cost", {})
    packet["compute_cost"]["critic_tokens"] = int(
        usage.get("total_tokens", 0)
    )

    parser_tokens = packet["compute_cost"].get("parser_tokens")
    parser_tokens = int(parser_tokens or 0)

    packet["compute_cost"]["total_llm_tokens"] = (
        parser_tokens
        + int(usage.get("total_tokens", 0))
    )

    (attempt_dir / "critic_summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    _write_json(
        attempt_dir / "result_packet.json",
        packet,
    )

    return packet, usage


def run_refinement_loop(
    spec,
    *,
    parser_usage=None,
    run_provenance=None,
    out_root: str | Path | None = None,
    max_attempts: int = 3,
    max_total_extra_iterations: int = 100,
    extension_preauthorized: bool = False,
    feedback_callback: FeedbackCallback | None = None,
    approval_callback: ApprovalCallback | None = None,
) -> dict:
    """Run a bounded verification-gated between-run refinement loop."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    if max_total_extra_iterations < 0:
        raise ValueError(
            "max_total_extra_iterations cannot be negative"
        )

    if out_root is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = (
            RUNS_ROOT
            / "refinement"
            / f"session_{timestamp}"
        )
    else:
        session_dir = Path(out_root)

    session_dir.mkdir(parents=True, exist_ok=True)

    base_max_iter = int(spec.simp.max_iter)
    extra_iterations_used = 0

    history: list[dict] = []

    total_repair_tokens = 0
    final_packet = None
    final_status = "unknown"

    for attempt in range(1, max_attempts + 1):

        print("\n" + "=" * 70)
        print(f"REFINEMENT ATTEMPT {attempt}/{max_attempts}")
        print("=" * 70)

        attempt_dir = session_dir / f"attempt_{attempt:03d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        executed_max_iter = (
            base_max_iter + extra_iterations_used
        )

        remaining_extra = max(
            0,
            max_total_extra_iterations
            - extra_iterations_used,
        )

        print(
            f"Frozen ProblemSpec max_iter: {base_max_iter}\n"
            f"Execution max_iter:          {executed_max_iter}\n"
            f"Remaining extra budget:      {remaining_extra}"
        )

        try:
            # Parser ran exactly once. Do not count its API usage again
            # on subsequent deterministic reruns.
            attempt_parser_usage = (
                parser_usage if attempt == 1 else None
            )

            packet = main_from_spec(
                spec,
                parser_usage=attempt_parser_usage,
                out_dir=str(attempt_dir),
                run_provenance=run_provenance,
                execution_overrides={
                    "max_iter": executed_max_iter,
                },
                run_critic=False,
            )

        except Exception as exc:
            error_record = {
                "status": "execution_error",
                "attempt": attempt,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

            _write_json(
                attempt_dir / "refinement_error.json",
                error_record,
            )

            final_status = "execution_error"

            summary = {
                "status": final_status,
                "session_directory": str(session_dir),
                "attempts": history,
                "error": error_record,
            }

            _write_json(
                session_dir / "refinement_summary.json",
                summary,
            )

            print(
                "\nRefinement stopped because the deterministic "
                "execution raised an exception."
            )

            return summary

        previous = [
            _compact_attempt(item)
            for item in history
        ]

        evaluation = build_evaluation_packet(
            packet,
            attempt=attempt,
            remaining_extra_iterations=remaining_extra,
            previous_attempts=previous,
        )

        # --------------------------------------------------------
        # Optional human feedback after seeing deterministic results
        # --------------------------------------------------------

        if (
            not evaluation.terminal_success
            and feedback_callback is not None
        ):
            feedback = feedback_callback(evaluation)

            evaluation = evaluation.model_copy(
                update={
                    "user_feedback": (
                        feedback.strip()
                        if feedback and feedback.strip()
                        else None
                    )
                }
            )

        _write_json(
            attempt_dir / "evaluation.json",
            evaluation,
        )

        record = {
            "attempt": attempt,
            "attempt_directory": str(attempt_dir),
            "evaluation": evaluation.model_dump(mode="json"),
            "proposal": None,
            "repair_agent_usage": None,
            "policy": None,
        }

        # --------------------------------------------------------
        # Successful terminal run
        # --------------------------------------------------------

        if evaluation.terminal_success:
            history.append(record)

            packet, critic_usage = _attach_final_critic(
                packet,
                attempt_dir=attempt_dir,
                history=history,
            )

            final_packet = packet
            final_status = "success"

            summary = {
                "status": final_status,
                "session_directory": str(session_dir),
                "attempt_count": len(history),
                "extra_iterations_used": extra_iterations_used,
                "repair_agent_tokens": total_repair_tokens,
                "final_critic_tokens": int(
                    critic_usage.get("total_tokens", 0)
                ),
                "attempts": history,
                "final_result_packet": str(
                    attempt_dir / "result_packet.json"
                ),
            }

            _write_json(
                session_dir / "refinement_summary.json",
                summary,
            )

            print("\nREFINEMENT COMPLETE: SUCCESS")
            print(
                f"Attempts: {len(history)} | "
                f"Extra iteration budget used: "
                f"{extra_iterations_used}"
            )

            return summary

        # --------------------------------------------------------
        # Ask LLM for one proposal
        # --------------------------------------------------------

        proposal, repair_usage = propose_repair(evaluation)

        total_repair_tokens += int(
            repair_usage.get("total_tokens", 0)
        )

        _write_json(
            attempt_dir / "repair_proposal.json",
            proposal,
        )

        # --------------------------------------------------------
        # Deterministic policy gate
        # --------------------------------------------------------

        decision = decide_repair(
            proposal,
            evaluation,
            extension_preauthorized=extension_preauthorized,
        )
        print("\n--- Deterministic Repair Policy ---")
        print(decision.model_dump_json(indent=2))

        # A non-preauthorized extension requires explicit approval.
        if (
            proposal.action == RepairAction.EXTEND_ITERATIONS
            and decision.allowed
            and decision.requires_human_approval
        ):
            approved = False

            if approval_callback is not None:
                approved = bool(
                    approval_callback(
                        proposal,
                        decision,
                        evaluation,
                    )
                )

            if approved:
                decision = decision.model_copy(
                    update={
                        "requires_human_approval": False,
                        "reason": (
                            decision.reason
                            + " Human approval received."
                        ),
                    }
                )
            else:
                decision = decision.model_copy(
                    update={
                        "allowed": False,
                        "stop": True,
                        "reason": (
                            decision.reason
                            + " Human approval was not received."
                        ),
                        "additional_iterations": 0,
                    }
                )

        _write_json(
            attempt_dir / "policy_decision.json",
            decision,
        )

        record["proposal"] = proposal.model_dump(mode="json")
        record["repair_agent_usage"] = repair_usage
        record["policy"] = decision.model_dump(mode="json")

        history.append(record)

        # --------------------------------------------------------
        # Stop conditions
        # --------------------------------------------------------

        if proposal.action == RepairAction.REQUEST_HUMAN_REVIEW:
            final_status = "human_review_required"
            break

        if decision.stop or not decision.allowed:
            final_status = "stopped_by_policy"
            break

        # --------------------------------------------------------
        # Apply the ONLY V1 automatic change:
        # execution iteration budget
        # --------------------------------------------------------

        extra_iterations_used += int(
            decision.additional_iterations
        )

        print(
            "\nPolicy approved rerun:"
            f" +{decision.additional_iterations} iterations"
        )

    else:
        final_status = "max_attempts_reached"

    summary = {
        "status": final_status,
        "session_directory": str(session_dir),
        "attempt_count": len(history),
        "extra_iterations_used": extra_iterations_used,
        "repair_agent_tokens": total_repair_tokens,
        "attempts": history,
        "final_result_packet": (
            None if final_packet is None else "available"
        ),
    }

    _write_json(
        session_dir / "refinement_summary.json",
        summary,
    )

    print(f"\nREFINEMENT STOPPED: {final_status}")

    return summary
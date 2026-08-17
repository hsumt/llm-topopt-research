"""Deterministically classify one completed TopOpt attempt."""

from __future__ import annotations

from project.refinement.schema import EvaluationPacket


def _failed_checks(validation: dict, severity: str) -> list[str]:
    """Return failed validation checks having exactly the requested severity."""
    failed: list[str] = []

    for name, check in validation.get("checks", {}).items():
        if not isinstance(check, dict):
            continue

        if (
            check.get("severity") == severity
            and check.get("passed") is False
        ):
            failed.append(name)

    return sorted(failed)


def build_evaluation_packet(
    result_packet: dict,
    *,
    attempt: int,
    remaining_extra_iterations: int,
    previous_attempts: list[dict] | None = None,
    user_feedback: str | None = None,
) -> EvaluationPacket:
    """Build the deterministic evidence packet used by refinement."""

    validation = result_packet.get("validation", {})
    final_result = result_packet.get("final_result", {})
    provenance = result_packet.get("interaction_provenance", {})
    verification = result_packet.get("numerical_verification_suite", {})

    critic_input = result_packet.get("what_claude_saw", {})
    run_config = critic_input.get("run_config", {})
    parsed_spec = critic_input.get("parsed_spec", {})

    hard_failed = _failed_checks(validation, "hard")
    quality_failed = _failed_checks(validation, "quality")

    design_converged = bool(
        final_result.get("design_converged", False)
    )

    continuation_complete = bool(
        final_result.get("continuation_complete", False)
    )

    requested_continuation_complete = bool(
        final_result.get("requested_continuation_complete", False)
    )

    # Make these failures explicit even if an older gate did not emit
    # the corresponding quality check.
    if (
        not design_converged
        and "design_convergence_status" not in quality_failed
    ):
        quality_failed.append("design_convergence_status")

    if (
        not requested_continuation_complete
        and "requested_continuation_incomplete" not in quality_failed
    ):
        quality_failed.append("requested_continuation_incomplete")

    quality_failed = sorted(set(quality_failed))

    semantic_status = (
        provenance
        .get("semantic_assurance", {})
        .get("status", "unknown")
    )

    original_max_iter = run_config.get("max_iter_requested")

    if original_max_iter is None:
        original_max_iter = (
            parsed_spec
            .get("simp", {})
            .get("max_iter")
        )

    if original_max_iter is None:
        original_max_iter = final_result.get("iterations", 1)

    execution_max_iter_budget = run_config.get(
        "max_iter_executed",
        original_max_iter,
    )

    actual_iterations = int(
        final_result.get("iterations", 1)
    )

    hard_validation_passed = bool(
        validation.get("passed", False)
    )

    verification_available = bool(
        verification.get("available", False)
    )
    verification_current = bool(
        verification.get("current", False)
    )
    verification_passed = bool(
        verification.get("passed", False)
    )

    semantic_ok = semantic_status in {
        "fully_explicit",
        "user_confirmed",
    }

    verification_ok = (
        verification_current
        and verification_passed
    )

    # This is deliberately stricter than validation["passed"] alone.
    #
    # Hard validation establishes numerical admissibility.
    # For refinement termination we additionally require:
    #   - design convergence;
    #   - completion of the requested continuation;
    #   - semantic assurance;
    #   - current numerical-verification evidence.
    terminal_success = bool(
        hard_validation_passed
        and not hard_failed
        and design_converged
        and requested_continuation_complete
        and semantic_ok
        and verification_ok
    )

    feedback = None
    if user_feedback is not None:
        stripped = user_feedback.strip()
        feedback = stripped if stripped else None

    return EvaluationPacket(
        attempt=attempt,
        hard_validation_passed=hard_validation_passed,
        hard_failed_checks=hard_failed,
        hard_failure_reasons=list(
            validation.get("failure_reasons", [])
        ),
        quality_failed_checks=quality_failed,
        quality_warnings=list(
            validation.get("quality_warnings", [])
        ),
        design_converged=design_converged,
        objective_plateau=bool(
            final_result.get("objective_plateau", False)
        ),
        continuation_complete=continuation_complete,
        requested_continuation_complete=(
            requested_continuation_complete
        ),
        semantic_assurance_status=semantic_status,
        numerical_verification_available=verification_available,
        numerical_verification_current=verification_current,
        numerical_verification_passed=verification_passed,
        original_max_iter=int(original_max_iter),
        execution_max_iter_budget=int(
            execution_max_iter_budget
        ),
        actual_iterations=actual_iterations,
        remaining_extra_iterations=int(
            remaining_extra_iterations
        ),
        terminal_success=terminal_success,
        user_feedback=feedback,
        previous_attempts=list(previous_attempts or []),
    )
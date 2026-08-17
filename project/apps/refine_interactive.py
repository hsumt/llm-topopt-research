"""Interactive verification-gated TopOpt refinement demo.

Open this file in VS Code and click Run Python File.
"""

from __future__ import annotations

from project.apps.interactive import main as interactive_main
from project.refinement.controller import run_refinement_loop


MAX_ATTEMPTS = 3
MAX_TOTAL_EXTRA_ITERATIONS = 100


def _feedback_callback(evaluation):
    print("\n--- Deterministic Evaluation ---")

    print(
        "Hard validation passed:",
        evaluation.hard_validation_passed,
    )

    print(
        "Hard failed checks:",
        evaluation.hard_failed_checks or "none",
    )

    print(
        "Quality/convergence issues:",
        evaluation.quality_failed_checks or "none",
    )

    print(
        "Design converged:",
        evaluation.design_converged,
    )

    print(
        "Requested continuation complete:",
        evaluation.requested_continuation_complete,
    )

    print(
        "Current iteration budget:",
        evaluation.execution_max_iter_budget,
    )

    print(
        "Remaining authorized extra iterations:",
        evaluation.remaining_extra_iterations,
    )

    feedback = input(
        "\nOptional feedback for the repair agent "
        "(press Enter to skip):\n> "
    ).strip()

    return feedback or None


def _approval_callback(
    proposal,
    decision,
    evaluation,
):
    print("\n--- Repair Requires Approval ---")
    print(proposal.model_dump_json(indent=2))
    print("\nPolicy:")
    print(decision.reason)

    response = input(
        "\nApprove this rerun? [y/N]\n> "
    ).strip().lower()

    return response in {
        "y",
        "yes",
        "approve",
        "approved",
    }


def _run_with_refinement(
    spec,
    parser_usage=None,
    run_provenance=None,
):
    print("\n" + "=" * 70)
    print("VERIFICATION-GATED REFINEMENT MODE")
    print("=" * 70)

    print(
        "\nThe engineering ProblemSpec will remain frozen.\n"
        "The repair agent cannot change loads, supports, volume "
        "fraction, material, filter radius, penalization, geometry, "
        "objective, or FEM physics.\n"
    )

    print(
        "The only V1 automatic repair is a bounded extension of "
        "the optimization iteration budget."
    )

    answer = input(
        f"\nPre-authorize up to "
        f"{MAX_TOTAL_EXTRA_ITERATIONS} total extra optimization "
        "iterations if the deterministic evaluator indicates "
        "unfinished convergence/continuation? [y/N]\n> "
    ).strip().lower()

    preauthorized = answer in {
        "y",
        "yes",
        "approve",
        "approved",
    }

    summary = run_refinement_loop(
        spec,
        parser_usage=parser_usage,
        run_provenance=run_provenance,
        max_attempts=MAX_ATTEMPTS,
        max_total_extra_iterations=(
            MAX_TOTAL_EXTRA_ITERATIONS
        ),
        extension_preauthorized=preauthorized,
        feedback_callback=_feedback_callback,
        approval_callback=_approval_callback,
    )

    print("\n" + "=" * 70)
    print("REFINEMENT SESSION FINISHED")
    print("=" * 70)

    print("Status:", summary["status"])
    print(
        "Session directory:",
        summary["session_directory"],
    )


if __name__ == "__main__":
    interactive_main(
        run_function=_run_with_refinement
    )
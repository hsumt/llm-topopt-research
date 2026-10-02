"""A provider failure must not become a request for missing engineering data."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from project import ai, workflow
from project.ai_errors import ModelCallError
from project.lbracket import reference_spec
from project.models import ProblemSpec, Resolution, Review, Session, Usage


def usage(step):
    return Usage(step=step, model="test")


def unavailable():
    return ModelCallError(
        code="missing_api_key", message="Claude is not configured for this app.",
        next_step="Supply the API key at container startup, then retry your request.",
        request_state="not_sent", retryable=False,
    )


class WorkflowFailureTests(unittest.TestCase):
    def test_parse_failure_is_not_a_missing_specification(self):
        with patch.object(workflow, "parse_problem", side_effect=unavailable()), \
             patch.object(workflow, "review_problem") as reviewer:
            failed = workflow.start_session("A detailed L-bracket specification", "drawing data")
        reviewer.assert_not_called()
        self.assertFalse(workflow.parse_succeeded(failed))
        self.assertFalse(workflow.ready_to_try(failed))
        self.assertEqual(workflow.deterministic_blockers(failed), [])
        self.assertEqual(failed.review, Review())
        self.assertEqual(failed.failure.stage, "parse")
        self.assertEqual(failed.failure.request_state, "not_sent")
        self.assertEqual(failed.original_request, "A detailed L-bracket specification")
        self.assertEqual(failed.context, "drawing data")

    def test_parser_retry_runs_parser_before_reviewer_and_keeps_audit(self):
        with patch.object(workflow, "parse_problem", side_effect=unavailable()):
            failed = workflow.start_session("original", "context")
        calls = []

        def parse(request, context):
            calls.append(("parse", request, context))
            return reference_spec(), usage("parse")

        def review(spec, **kwargs):
            calls.append(("review", spec.name, kwargs["original_request"]))
            return Review(), usage("review")

        with patch.object(workflow, "parse_problem", side_effect=parse), \
             patch.object(workflow, "review_problem", side_effect=review):
            recovered = workflow.retry_review(failed)
        self.assertEqual([call[0] for call in calls], ["parse", "review"])
        self.assertEqual(calls[0], ("parse", "original", "context"))
        self.assertTrue(workflow.ready_to_try(recovered))
        self.assertIsNone(recovered.failure)
        self.assertTrue(recovered.parse_completed)
        self.assertEqual(recovered.errors, failed.errors)

    def test_legacy_unparsed_session_retries_parser(self):
        legacy = Session(original_request="old saved request", spec=ProblemSpec(name="Unparsed request"),
                         review=Review(), errors=["The parse step failed."])
        self.assertFalse(workflow.parse_succeeded(legacy))
        self.assertEqual(workflow.deterministic_blockers(legacy), [])
        with patch.object(workflow, "parse_problem", return_value=(reference_spec(), usage("parse"))) as parser, \
             patch.object(workflow, "review_problem", return_value=(Review(), usage("review"))):
            recovered = workflow.retry_failure(legacy)
        parser.assert_called_once_with("old saved request", None)
        self.assertEqual(recovered.errors, legacy.errors)
        self.assertTrue(workflow.ready_to_try(recovered))

    def test_reference_session_without_model_parse_remains_supported(self):
        benchmark = Session(original_request="reference", spec=reference_spec(), review=Review())
        self.assertTrue(workflow.parse_succeeded(benchmark))
        self.assertTrue(workflow.ready_to_try(benchmark))

    def test_review_failure_preserves_parse_and_retry_does_not_reparse(self):
        with patch.object(workflow, "parse_problem", return_value=(reference_spec(), usage("parse"))), \
             patch.object(workflow, "review_problem", side_effect=RuntimeError("PRIVATE_RESPONSE")):
            failed = workflow.start_session("reference")
        self.assertTrue(workflow.parse_succeeded(failed))
        self.assertEqual(failed.failure.stage, "review")
        self.assertFalse(workflow.ready_to_try(failed))
        self.assertNotIn("PRIVATE_RESPONSE", failed.model_dump_json())
        with patch.object(workflow, "parse_problem") as parser, \
             patch.object(workflow, "review_problem", return_value=(Review(), usage("review"))):
            recovered = workflow.retry_failure(failed)
        parser.assert_not_called()
        self.assertEqual(recovered.spec, failed.spec)
        self.assertEqual(recovered.errors, failed.errors)
        self.assertIsNone(recovered.failure)
        self.assertTrue(workflow.ready_to_try(recovered))

    def test_failed_answers_are_retained_and_retried_once(self):
        original = Session(original_request="reference", spec=reference_spec(), review=Review())
        answers = {"name": "approved design"}
        with patch.object(workflow, "resolve_answers", side_effect=RuntimeError("private")):
            failed = workflow.answer_questions(original, answers)
        self.assertEqual(failed.failure.stage, "resolve")
        self.assertEqual(failed.pending_answers, answers)
        self.assertEqual(failed.revisions, [])
        self.assertEqual(failed.spec, original.spec)
        resolution = Resolution(updates=[{"path": ["name"], "value": "approved design"}])
        with patch.object(workflow, "parse_problem") as parser, \
             patch.object(workflow, "resolve_answers", return_value=(resolution, usage("resolve"))) as resolver, \
             patch.object(workflow, "review_problem", return_value=(Review(), usage("review"))):
            recovered = workflow.retry_failure(failed)
        parser.assert_not_called()
        self.assertEqual(resolver.call_args.args[2], answers)
        self.assertEqual(recovered.spec.name, "approved design")
        self.assertEqual(recovered.pending_answers, {})
        self.assertEqual(len(recovered.revisions), 1)
        self.assertEqual(recovered.revisions[0].answers, answers)
        self.assertEqual(recovered.errors, failed.errors)
        self.assertIsNone(recovered.failure)
        self.assertTrue(workflow.ready_to_try(recovered))

    def test_failed_review_after_answer_does_not_reapply_resolution(self):
        original = Session(original_request="reference", spec=reference_spec(), review=Review())
        resolution = Resolution(updates=[{"path": ["name"], "value": "accepted update"}])
        with patch.object(workflow, "resolve_answers", return_value=(resolution, usage("resolve"))), \
             patch.object(workflow, "review_problem", side_effect=RuntimeError("private")):
            failed = workflow.answer_questions(original, {"name": "accepted update"})
        self.assertEqual(failed.failure.stage, "review")
        self.assertEqual(failed.spec.name, "accepted update")
        self.assertEqual(len(failed.revisions), 1)
        self.assertEqual(failed.pending_answers, {})
        with patch.object(workflow, "resolve_answers") as resolver, \
             patch.object(workflow, "review_problem", return_value=(Review(), usage("review"))):
            recovered = workflow.retry_failure(failed)
        resolver.assert_not_called()
        self.assertEqual(recovered.spec, failed.spec)
        self.assertEqual(recovered.revisions, failed.revisions)
        self.assertTrue(workflow.ready_to_try(recovered))

    def test_invalid_patch_is_atomic_and_marks_received_response(self):
        original = Session(original_request="reference", spec=reference_spec(), review=Review())
        resolution = Resolution(updates=[
            {"path": ["name"], "value": "must not be kept"},
            {"path": ["unknown_field"], "value": "invalid"},
        ])
        with patch.object(workflow, "resolve_answers", return_value=(resolution, usage("resolve"))), \
             patch.object(workflow, "review_problem") as reviewer:
            failed = workflow.answer_questions(original, {"request": "update"})
        reviewer.assert_not_called()
        self.assertEqual(failed.failure.code, "invalid_schema")
        self.assertEqual(failed.failure.request_state, "response_received")
        self.assertEqual(failed.spec, original.spec)
        self.assertEqual(failed.pending_answers, {"request": "update"})
        self.assertEqual(failed.revisions, [])

    def test_answering_unparsed_session_cannot_skip_parse(self):
        with patch.object(workflow, "parse_problem", side_effect=unavailable()):
            failed = workflow.start_session("original")
        with patch.object(workflow, "resolve_answers") as resolver:
            result = workflow.answer_questions(failed, {"answer": "data"})
        resolver.assert_not_called()
        self.assertEqual(result, failed)
        self.assertFalse(workflow.ready_to_try(result))

    def test_invalid_cached_patch_retry_fetches_new_changes(self):
        original = Session(original_request="reference", spec=reference_spec(), review=Review())
        answers = {"name": "accepted update"}
        invalid = Resolution(updates=[{"path": ["unknown_field"], "value": "invalid"}])
        valid = Resolution(updates=[{"path": ["name"], "value": "accepted update"}])

        def response(resolution):
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=resolution.model_dump_json())],
                stop_reason="end_turn", usage=SimpleNamespace(input_tokens=20, output_tokens=10),
            )

        with tempfile.TemporaryDirectory() as cache_dir, \
             patch.object(ai, "CACHE_DIR", Path(cache_dir)), \
             patch.object(ai, "USE_CACHE", True), \
             patch.object(ai, "_client") as client, \
             patch.object(workflow, "review_problem", return_value=(Review(), usage("review"))):
            create = client.return_value.messages.create
            create.return_value = response(invalid)
            first = workflow.answer_questions(original, answers)
            self.assertEqual(first.failure.request_state, "response_received")
            self.assertEqual(create.call_count, 1)

            # The invalid patch passes the Resolution schema and was cached.
            cached = workflow.answer_questions(original, answers)
            self.assertEqual(cached.failure.request_state, "not_sent")
            self.assertIn("saved Claude response", cached.failure.message)
            self.assertEqual(create.call_count, 1)

            # A network failure during the first retry must not cause later
            # retries to fall back to the known-bad cached patch again.
            create.side_effect = ConnectionError("private network detail")
            disconnected = workflow.retry_failure(cached)
            self.assertEqual(disconnected.failure.code, "connection_failed")
            self.assertEqual(create.call_count, 2)
            self.assertEqual(disconnected.pending_answers, answers)

            create.side_effect = None
            create.return_value = response(valid)
            recovered = workflow.retry_failure(disconnected)
            self.assertEqual(create.call_count, 3)
            self.assertEqual(recovered.spec.name, "accepted update")
            self.assertEqual(recovered.revisions[0].answers, answers)
            self.assertEqual(recovered.errors, disconnected.errors)
            self.assertIsNone(recovered.failure)
            self.assertTrue(workflow.ready_to_try(recovered))

            # The valid replacement is now what an identical fresh request sees.
            cached_valid = workflow.answer_questions(original, answers)
            self.assertEqual(create.call_count, 3)
            self.assertTrue(cached_valid.usage[0].cache_hit)
            self.assertEqual(cached_valid.spec.name, "accepted update")


if __name__ == "__main__":
    unittest.main()

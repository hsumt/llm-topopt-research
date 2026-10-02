"""UI regression checks for provider failures and engineering questions."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from project import execution, workflow
from project.ai_errors import ModelCallError
from project.examples import EXAMPLES
from project.lbracket import combine_review, reference_spec
from project.models import Review, Session, Usage


@unittest.skipUnless(importlib.util.find_spec("streamlit"), "UI checks run in the Docker image")
class AppFailureGuidanceTests(unittest.TestCase):
    def app(self):
        from streamlit.testing.v1 import AppTest
        return AppTest.from_file(str(Path(__file__).resolve().parents[1] / "project/app.py"))

    def button(self, app, label):
        return next(button for button in app.button if button.label == label)

    def text(self, app):
        return "\n".join(str(item.value) for kind in ("markdown", "caption", "error", "warning", "info", "subheader")
                         for item in getattr(app, kind))

    def missing_key(self):
        return ModelCallError(
            code="missing_api_key", message="Anthropic is not configured in this app instance.",
            next_step="Supply ANTHROPIC_API_KEY to the Docker service, then retry parsing.",
            request_state="not_sent", retryable=False,
        )

    def test_failed_build_explains_setup_and_retries_parser_without_bogus_questions(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(execution, "ARTIFACTS", Path(directory)), \
             patch.object(workflow, "parse_problem", side_effect=[self.missing_key(),
                          (reference_spec(), Usage(step="parse", model="mock"))]) as parser, \
             patch.object(workflow, "review_problem", return_value=(Review(), Usage(step="review", model="mock"))) as critic:
            app = self.app().run(timeout=30)
            brief = next(iter(EXAMPLES.values()))
            next(t for t in app.text_area if t.label == "Describe the engineering problem").set_value(brief["request"])
            next(t for t in app.text_area if t.label == "Additional context").set_value(brief["context"])
            self.button(app, "Build formulation").click().run(timeout=30)
            self.assertFalse(app.exception)
            text = self.text(app)
            self.assertIn("No request was sent to Anthropic", text)
            self.assertNotIn("The material has not been identified", text)
            self.assertNotIn("The load is missing", text)
            self.assertNotIn("Current formulation", text)
            self.assertEqual(next(t for t in app.text_area if t.label == "Your engineering brief").value, brief["request"])
            self.assertTrue(any("read -r -s" in block.value for block in app.code))
            critic.assert_not_called()
            self.button(app, "Retry parsing").click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(parser.call_count, 2)
            critic.assert_called_once()
            self.assertTrue(app.session_state.session.parse_completed)
            self.assertIsNone(app.session_state.session.failure)

    def test_legacy_unparsed_session_has_parser_retry_and_keeps_original(self):
        from project.models import ProblemSpec
        app = self.app().run(timeout=30)
        app.session_state.session = Session(original_request="Original saved brief", spec=ProblemSpec(name="Unparsed request"), review=Review())
        app.run(timeout=30)
        self.assertFalse(app.exception)
        self.assertIn("does not contain a specific error diagnosis", self.text(app))
        self.assertIsNotNone(self.button(app, "Retry parsing"))
        self.assertFalse(any(b.label == "Retry formulation review" for b in app.button))

    def test_actual_missing_decisions_display_examples_without_prefilling_answers(self):
        spec = reference_spec()
        del spec.geometry.parameters["load_patch"]
        spec.boundary_conditions[0].kind = None
        spec.boundary_conditions[0].components = []
        spec.optimization.constraints[0].region = None
        session = Session(original_request="Preliminary brief", spec=spec, review=combine_review(spec, Review()), parse_completed=True)
        app = self.app().run(timeout=30)
        app.session_state.session = session
        app.run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.text_input), 3)
        self.assertTrue(all(field.value == "" for field in app.text_input))
        self.assertEqual(sum("Example answer" in item.value for item in app.info), 3)
        self.assertFalse(any(b.label == "Run 3D L-bracket" for b in app.button))
        self.assertEqual(app.session_state.session.spec, spec)


if __name__ == "__main__":
    unittest.main()

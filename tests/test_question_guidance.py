"""Clarification guidance must be useful without turning examples into intent."""
import unittest

from project.lbracket import assess_spec, combine_review, reference_spec, starter_spec
from project.models import BoundaryCondition, Review, Value


def preliminary_brief_spec():
    """The starter brief already supplies all but three physical decisions."""
    spec = reference_spec()
    del spec.geometry.parameters["load_patch"]
    spec.boundary_conditions = [BoundaryCondition(name="Mount to rigid frame", location="top_arm")]
    spec.loads[0].kind = None
    spec.loads[0].location = None
    spec.optimization.constraints[0].region = None
    return spec


class QuestionGuidanceTests(unittest.TestCase):
    def test_preliminary_brief_asks_three_actionable_questions(self):
        assessment = assess_spec(preliminary_brief_spec())
        self.assertFalse(assessment.ready)
        self.assertEqual(len(assessment.questions), 3)
        questions = {q.key: q for q in assessment.questions}

        load = questions["solver_dimensions"]
        self.assertEqual(set(load.issue_keys), {"solver_dimensions", "solver_load_model"})
        self.assertIn("height with units", load.prompt)
        self.assertIn("uniformly", load.prompt)
        self.assertIn("4 mm", load.example_answer)
        self.assertIn("12 mm thickness", load.example_answer)

        support = questions["solver_support"]
        self.assertIn("motion", support.prompt)
        self.assertIn("entire top face", support.example_answer)
        self.assertIn("zero", support.example_answer)
        self.assertIn("x, y and z", support.example_answer)

        volume = questions["solver_volume_reference"]
        self.assertIn("75%", volume.example_answer)
        self.assertIn("before subtracting", volume.example_answer)
        self.assertTrue(all(q.example_answer for q in assessment.questions))

    def test_examples_never_fill_missing_inputs_or_create_config(self):
        spec = preliminary_brief_spec()
        before = spec.model_dump()
        for _ in range(2):
            assessment = assess_spec(spec)
            self.assertIsNone(assessment.config)
            self.assertEqual(spec.model_dump(), before)
            self.assertNotIn("load_patch", spec.geometry.parameters)
            self.assertIsNone(spec.boundary_conditions[0].kind)
            self.assertIsNone(spec.optimization.constraints[0].region)
        review = combine_review(spec, Review())
        self.assertEqual(len(review.questions), 3)
        self.assertEqual(spec.model_dump(), before)

    def test_only_missing_geometry_field_is_requested(self):
        spec = reference_spec()
        del spec.geometry.parameters["hole_radius"]
        assessment = assess_spec(spec)
        self.assertEqual(len(assessment.questions), 1)
        question = assessment.questions[0]
        self.assertIn("initial hole radius", question.prompt)
        self.assertNotIn("outer x length", question.prompt)
        self.assertNotIn("thickness", question.prompt)
        self.assertNotIn("hole centers", question.prompt)
        self.assertIn("mm", question.example_answer)
        self.assertIsNone(assessment.config)

    def test_examples_respect_already_stated_quantities(self):
        spec = preliminary_brief_spec()
        spec.geometry.parameters["thickness"] = Value(value=24, unit="mm")
        spec.geometry.parameters["load_patch"] = Value(value=6, unit="mm")
        spec.optimization.constraints[0].limit = Value(value=60, unit="%")
        questions = {q.key: q for q in assess_spec(spec).questions}
        self.assertIn("6 mm", questions["solver_load_model"].example_answer)
        self.assertIn("24 mm", questions["solver_load_model"].example_answer)
        self.assertIn("60%", questions["solver_volume_reference"].example_answer)
        self.assertNotIn("75%", questions["solver_volume_reference"].example_answer)

    def test_missing_identified_data_does_not_become_an_example_guess(self):
        spec = reference_spec()
        spec.materials[0].properties = {}
        del spec.geometry.parameters["load_patch"]
        spec.geometry.authoritative_source = "Drawing A, not yet supplied"
        assessment = assess_spec(spec)
        self.assertFalse(assessment.ready)
        self.assertEqual(assessment.questions, [])
        self.assertEqual({i.kind for i in assessment.issues}, {"data"})

    def test_unsupported_mount_and_load_remain_blocking(self):
        spec = reference_spec()
        spec.boundary_conditions[0].kind = "pinned"
        spec.loads[0].kind = "point_force"
        assessment = assess_spec(spec)
        self.assertFalse(assessment.ready)
        self.assertEqual(assessment.questions, [])
        self.assertEqual({i.kind for i in assessment.issues}, {"unsupported"})
        self.assertTrue(all("implementation" in i.description for i in assessment.issues))

    def test_question_round_is_bounded(self):
        self.assertLessEqual(len(assess_spec(starter_spec()).questions), 4)


if __name__ == "__main__":
    unittest.main()

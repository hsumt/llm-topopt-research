"""Behavioral checks for the deterministic pre-solve contract (no API/FEM)."""
import sys
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from project import ai, workflow
from project.lbracket import assess_spec, combine_review, reference_spec, starter_spec
from project.models import Issue, Manufacturing, ProblemSpec, Question, Region, Resolution, Review, Session, Usage, Value
from project.solver.lbracket3d.config import reference_config


def usage(step):
    return Usage(step=step, model="test")


class AdapterTests(unittest.TestCase):
    def test_explicit_reference_matches_source_contract(self):
        assessment = assess_spec(reference_spec())
        self.assertEqual(assessment.issues, [])
        self.assertEqual(assessment.config, reference_config())
        self.assertTrue(assessment.ready)

    def test_empty_critic_cannot_authorize_incomplete_spec(self):
        session = Session(original_request="optimize a 3D L bracket", spec=starter_spec(), review=Review())
        self.assertFalse(workflow.ready_to_try(session))
        merged = combine_review(session.spec, Review())
        self.assertTrue(merged.issues)
        self.assertGreater(len(merged.questions), 0)
        self.assertLessEqual(len(merged.questions), 4)

    def test_missing_material_values_are_data_not_nominal_questions(self):
        spec = reference_spec()
        spec.materials[0].properties = {}
        assessment = assess_spec(spec)
        self.assertIsNone(assessment.config)
        self.assertEqual({i.kind for i in assessment.issues}, {"data"})
        self.assertEqual(assessment.questions, [])

    def test_model_cannot_turn_missing_modulus_into_nominal_guess_question(self):
        spec = reference_spec(); spec.materials[0].properties = {}
        review = Review(issues=[Issue(key="material_E", field="materials.0.properties.E", description="Missing Young's modulus")],
                        questions=[Question(key="guess_E", issue_keys=["material_E"], prompt="Which nominal modulus should I assume?", why="fill field")])
        merged = combine_review(spec, review)
        self.assertEqual(merged.questions, [])
        self.assertEqual({i.kind for i in merged.issues}, {"data"})

    def test_units_convert_without_changing_problem(self):
        spec = reference_spec()
        for name, value in spec.geometry.parameters.items():
            if name == "hole_centers":
                value.value = [[1000*x, 1000*y] for x,y in value.value]
            else:
                value.value *= 1000
            value.unit = "mm"
        spec.materials[0].properties["E"] = Value(value=120, unit="GPa")
        spec.loads[0].magnitude = Value(value=5, unit="kN")
        spec.loads[0].direction = "-y"
        spec.optimization.constraints[0].limit = Value(value=75, unit="%")
        spec.optimization.constraints[1].limit = Value(value=116, unit="MPa")
        assessment = assess_spec(spec)
        self.assertEqual(assessment.issues, [])
        self.assertEqual(assessment.config, reference_config())

    def test_volume_only_is_explicit_and_changes_config(self):
        spec = reference_spec()
        spec.optimization.constraints = spec.optimization.constraints[:1]
        spec.optimization.stress_requirement = None
        self.assertFalse(assess_spec(spec).ready)
        spec.optimization.stress_requirement = "volume_only"
        self.assertTrue(assess_spec(spec).ready)
        self.assertIsNone(assess_spec(spec).config.stress_limit_pa)

    def test_never_reinterpret_yield_as_pnorm(self):
        spec = reference_spec()
        spec.optimization.constraints[1].quantity = "yield_stress"
        assessment = assess_spec(spec)
        self.assertIsNone(assessment.config)
        self.assertTrue(any(i.kind == "unsupported" for i in assessment.issues))

    def test_unsupported_physics_and_requirements_remain_blocking(self):
        variants = []
        spec = reference_spec(); spec.geometry.hole_role = "preserved_void"; variants.append(spec)
        spec = reference_spec(); spec.loads.append(spec.loads[0].model_copy()); variants.append(spec)
        spec = reference_spec(); spec.manufacturing = Manufacturing(process="CNC milling"); variants.append(spec)
        spec = reference_spec(); spec.geometry.regions.append(Region(name="mounting_holes", description="keep solid", role="protected")); variants.append(spec)
        spec = reference_spec(); spec.physics.regime = "nonlinear_dynamic"; variants.append(spec)
        spec = reference_spec(); spec.optimization.constraints[0].region = "bounding_box"; variants.append(spec)
        for variant in variants:
            with self.subTest(variant=variant.name):
                assessment = assess_spec(variant)
                self.assertFalse(assessment.ready)
                self.assertTrue(any(i.kind == "unsupported" for i in assessment.issues))

    def test_no_defaults_for_missing_geometry_or_force(self):
        spec = reference_spec()
        del spec.geometry.parameters["thickness"]
        self.assertFalse(assess_spec(spec).ready)
        spec = reference_spec(); spec.loads = []
        self.assertFalse(assess_spec(spec).ready)
        spec = reference_spec(); spec.loads[0].magnitude.unit = None
        self.assertFalse(assess_spec(spec).ready)

    def test_invalid_mesh_is_data_blocker(self):
        spec = reference_spec()
        spec.geometry.parameters["thickness"] = Value(value=1, unit="mm")
        assessment = assess_spec(spec)
        self.assertFalse(assessment.ready)
        self.assertTrue(any(i.kind == "data" for i in assessment.issues))

    def test_identified_missing_drawing_stays_data_without_repeat_question(self):
        spec = reference_spec()
        spec.geometry.parameters = {}
        spec.geometry.authoritative_source = "Engineer identified drawing A, not yet supplied"
        assessment = assess_spec(spec)
        self.assertFalse(assessment.ready)
        self.assertEqual(assessment.questions, [])
        self.assertEqual({i.kind for i in assessment.issues}, {"data"})


class ResolutionTests(unittest.TestCase):
    def test_add_new_value_map_key(self):
        spec = reference_spec()
        del spec.materials[0].properties["E"]
        result = workflow.apply_resolution(spec, Resolution(updates=[{
            "path": ["materials", 0, "properties", "E"], "value": {"value": 120, "unit": "GPa"},
        }]))
        self.assertTrue(assess_spec(result).ready)
        self.assertNotIn("E", spec.materials[0].properties)

    def test_add_value_key_still_requires_typed_value(self):
        spec = reference_spec()
        del spec.geometry.parameters["thickness"]
        with self.assertRaises(ValidationError):
            workflow.apply_resolution(spec, Resolution(updates=[{
                "path": ["geometry", "parameters", "thickness"], "value": 12,
            }]))

    def test_bad_index_or_unknown_field_rejected_atomically(self):
        spec = reference_spec()
        for path in (["loads", -1, "name"], ["loads", 8, "name"], ["secret"], ["physics", "undeclared"]):
            with self.subTest(path=path), self.assertRaises(ValueError):
                workflow.apply_resolution(spec, Resolution(updates=[
                    {"path": ["name"], "value": "changed"}, {"path": path, "value": "bad"},
                ]))
            self.assertEqual(spec.name, "3D L-bracket with five initial holes")

    def test_patch_can_invalidate_previously_ready_config(self):
        spec = workflow.apply_resolution(reference_spec(), Resolution(updates=[
            {"path": ["loads", 0, "kind"], "value": "point_force"},
        ]))
        session = Session(original_request="reference", spec=spec, review=Review())
        self.assertFalse(workflow.ready_to_try(session))

    def test_invalid_physics_default_was_fixed(self):
        data = reference_spec().model_dump()
        data["physics"] = {}
        spec = ProblemSpec.model_validate(data)
        self.assertEqual(ProblemSpec.model_validate(spec.model_dump()).physics.family, "unknown")
        self.assertFalse(assess_spec(spec).ready)


class WorkflowTests(unittest.TestCase):
    def test_empty_live_review_still_generates_questions(self):
        with patch.object(workflow, "parse_problem", return_value=(starter_spec(), usage("parse"))), \
             patch.object(workflow, "review_problem", return_value=(Review(), usage("review"))) as reviewer:
            session = workflow.start_session("my incomplete request", "governing context")
        self.assertFalse(workflow.ready_to_try(session))
        self.assertEqual(len(session.review.questions), 4)
        self.assertEqual(reviewer.call_args.kwargs["original_request"], "my incomplete request")
        self.assertEqual(reviewer.call_args.kwargs["context"], "governing context")

    def test_reviewer_and_resolver_get_original_and_prior_answers(self):
        session = Session(original_request="original intent", context="source", spec=reference_spec(), review=Review())
        resolution = Resolution(updates=[{"path": ["name"], "value": "renamed"}])
        with patch.object(workflow, "resolve_answers", return_value=(resolution, usage("resolve"))) as resolver, \
             patch.object(workflow, "review_problem", return_value=(Review(), usage("review"))) as reviewer:
            result = workflow.answer_questions(session, {"name": "renamed"})
        self.assertEqual(reviewer.call_args.kwargs["original_request"], "original intent")
        self.assertEqual(reviewer.call_args.kwargs["revisions"][-1].answers, {"name": "renamed"})
        self.assertEqual(resolver.call_args.kwargs["context"], "source")
        self.assertEqual(result.revisions[-1].answers, {"name": "renamed"})
        self.assertTrue(workflow.ready_to_try(result))

    def test_provider_error_is_sanitized_and_preserves_spec(self):
        session = Session(original_request="original", spec=reference_spec(), review=Review())
        with patch.object(workflow, "resolve_answers", side_effect=RuntimeError("SENSITIVE_PROVIDER_CONTENT")):
            result = workflow.answer_questions(session, {"x": "answer"})
        self.assertEqual(result.spec, session.spec)
        self.assertNotIn("SENSITIVE_PROVIDER_CONTENT", result.model_dump_json())
        self.assertTrue(result.errors)
        self.assertFalse(workflow.ready_to_try(result))

    def test_provider_payload_includes_context_and_prior_rounds(self):
        with patch.object(ai, "_call_model", return_value=(Review(), usage("review"))) as caller:
            ai.review_problem(reference_spec(), original_request="intent", context="drawing data", revisions=[])
        self.assertEqual(caller.call_args.kwargs["payload"]["original_request"], "intent")
        self.assertEqual(caller.call_args.kwargs["payload"]["context"], "drawing data")
        self.assertIn("prior_rounds", caller.call_args.kwargs["payload"])

    def test_review_retry_clears_transient_blocker_and_retains_error_audit(self):
        session = Session(original_request="reference", spec=reference_spec(), review=Review())
        with patch.object(workflow, "review_problem", side_effect=RuntimeError("private")):
            failed = workflow.retry_review(session)
        self.assertFalse(workflow.ready_to_try(failed))
        with patch.object(workflow, "review_problem", return_value=(Review(), usage("review"))):
            recovered = workflow.retry_review(failed)
        self.assertTrue(workflow.ready_to_try(recovered))
        self.assertEqual(recovered.errors, failed.errors)
        self.assertEqual(recovered.spec, session.spec)

    def test_deterministic_modules_do_not_import_fem_or_anthropic(self):
        for module in ("dolfinx", "petsc4py", "mpi4py", "anthropic"):
            self.assertNotIn(module, sys.modules)


if __name__ == "__main__":
    unittest.main()

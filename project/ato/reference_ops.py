"""Human-written reference operations -- a control, deliberately NOT registered.

These three edits are what a person who already knew the answer would write to
reach the 2.5D capability. They are kept out of ``grammar.OPERATIONS`` on purpose:
if they were registered, the hypothesis stage would find them by lookup and the
synthesis stage would never be exercised, which is precisely the bias this
architecture is meant to remove.

Their value is as a control. After a run in which the system synthesized its own
operation for the same purpose, compare the two: same slot? same tier? same
parameter, same bounds, same apply semantics? Divergence is a finding about the
synthesis stage, not about the physics.

To use them as a baseline instead of synthesis, register them explicitly:

    from project.ato.grammar import register_synthesized
    from project.ato.reference_ops import REFERENCE_OPERATIONS
    for op in REFERENCE_OPERATIONS.values():
        register_synthesized(op)
"""
from __future__ import annotations

from project.ato.grammar import EditOperation, _pos_float, _pos_int


def _apply_promote_model(spec, prm):
    spec.geometry.model = "extruded_3d"
    spec.geometry.n_layers = int(prm["n_layers"])


def _apply_transverse_load(spec, prm):
    from project.topopt.stress.problem import LoadCase

    if any(lc.direction == "z" for lc in spec.load_cases):
        for lc in spec.load_cases:
            if lc.direction == "z":
                lc.magnitude = float(prm["magnitude"])
        return
    spec.load_cases = list(spec.load_cases) + [
        LoadCase(magnitude=float(prm["magnitude"]), direction="z", name="transverse")
    ]


def _apply_thickness(spec, prm):
    spec.geometry.thickness = float(prm["thickness"])



REFERENCE_OPERATIONS = {op.name: op for op in [
        EditOperation(
            name="promote_analysis_model",
            slot="discretisation (analysis model)",
            description=(
                "Replace the 2-D plane-stress model with the 2.5D extruded solid model: "
                "the same in-plane cell grid swept through the thickness into n_layers of "
                "hex elements, with density held constant through the thickness. Every "
                "node gains a third displacement component, so an out-of-plane load case "
                "becomes expressible at all. The design variable count is unchanged and "
                "the in-plane response is within 0.3 percent of plane stress (measured). "
                "This edit changes no requirement and no geometry -- only what the model "
                "is capable of being asked. Few layers UNDERSTATE transverse stress "
                "(trilinear hexes shear-lock in bending), so use at least 4 for a "
                "transverse verdict."
            ),
            tier="T1",
            boundedness="BOUND",
            parameter_source=(
                "the layer count follows the transverse-stress convergence study in "
                "project/topopt/stress/verify_extrusion.py, not the anomaly"
            ),
            parameters={"n_layers": "int, 1..16, element layers through the thickness"},
            apply=_apply_promote_model,
            validate=lambda d: _pos_int(d, "n_layers", 1, 16),
        ),
        EditOperation(
            name="add_transverse_load_case",
            slot="load cases",
            description=(
                "Add an out-of-plane (z) load case at the arm tip, recording a service "
                "load the engineer knows the part sees but did not state. REQUIRES the "
                "extruded model: applied to a plane-stress model the analysis raises "
                "ModelCannotRepresent, because a plane-stress node has no z degree of "
                "freedom for the load to act on. It therefore has a verdict only in "
                "combination with promote_analysis_model."
            ),
            tier="T2",
            boundedness="UNBOUND",
            parameter_source=(
                "nothing in the specification or the anomaly localisation determines the "
                "magnitude of an omitted load; only the engineer's service envelope does"
            ),
            parameters={"magnitude": "float, 1..5000 N, out-of-plane tip load"},
            apply=_apply_transverse_load,
            validate=lambda d: _pos_float(d, "magnitude", 1.0, 5000.0),
        ),
        EditOperation(
            name="increase_thickness",
            slot="design domain (geometry)",
            description=(
                "Increase the out-of-plane thickness of the part. Out-of-plane bending "
                "stiffness scales with the cube of thickness, so this is the dominant "
                "lever on a transverse-load anomaly. It asserts a part geometry the "
                "engineer never stated and may contradict a mandated stock gauge, so it "
                "is reported and never applied on the system's own authority."
            ),
            tier="T3",
            boundedness="UNBOUND",
            parameter_source=(
                "the specification does not determine the thickness; a value that clears "
                "the anomaly is not evidence that it is the right value"
            ),
            parameters={"thickness": "float, 0.5..50 mm"},
            apply=_apply_thickness,
            validate=lambda d: _pos_float(d, "thickness", 0.5, 50.0),
        ),
]}

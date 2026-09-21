"""Declared design intent -- an input, not a result.

The T2/T3 distinction is not computable from the specification alone. Whether an
edit *reveals* a requirement the engineer already held or *alters* what they asked
for is a question about intent, and intent lives outside the numerics. This module
is where it is stated, explicitly and in one place, so that:

* an adequacy predicate can compare the specification against what the part is
  actually required to do, rather than inferring requirements from the very
  specification under suspicion;
* the authority stage has something to rule against;
* every tier assignment is auditable back to a written requirement.

Nothing here is derived from any run. Editing this file changes what the system
considers adequate, which is exactly the point: it is the engineer's statement of
the problem behind the problem statement.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

AXIS_MEANING = {
    "x": "in-plane, along the horizontal arm",
    "y": "in-plane, transverse to the horizontal arm (the stated service load)",
    "z": "out of plane, normal to the bracket face",
}


@dataclass
class DesignIntent:
    """What the part is required to do, independent of how it was specified."""

    required_load_axes: tuple = ("y",)
    """Axes along which the part must carry load. A required axis absent from the
    specification's load cases is a specification gap, whatever the stress field says."""

    axis_rationale: dict = field(default_factory=dict)
    """Why each axis is required -- free text, quoted verbatim in reports."""

    fixed_geometry: tuple = ()
    """Geometric quantities the engineer states are NOT free to change (e.g.
    ``("thickness",)`` when a stock plate gauge is mandated). An edit touching one
    of these is intent-altering however reasonable it looks."""

    mutable_geometry: tuple = ()
    """Geometric quantities the engineer explicitly permits changing."""

    stress_allowable_is_firm: bool = True
    """Whether the stated allowable may be renegotiated. If firm, relaxing it is
    intent-altering by definition rather than by the grammar's default guess."""

    mass_budget_is_firm: bool = True
    notes: str = ""
    provenance: str = "hand-written by the engineer"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["required_load_axes"] = list(self.required_load_axes)
        d["fixed_geometry"] = list(self.fixed_geometry)
        d["mutable_geometry"] = list(self.mutable_geometry)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DesignIntent":
        return cls(
            required_load_axes=tuple(d.get("required_load_axes", ("y",))),
            axis_rationale=dict(d.get("axis_rationale", {})),
            fixed_geometry=tuple(d.get("fixed_geometry", ())),
            mutable_geometry=tuple(d.get("mutable_geometry", ())),
            stress_allowable_is_firm=bool(d.get("stress_allowable_is_firm", True)),
            mass_budget_is_firm=bool(d.get("mass_budget_is_firm", True)),
            notes=str(d.get("notes", "")),
            provenance=str(d.get("provenance", "unstated")),
        )

    @classmethod
    def load(cls, path) -> "DesignIntent":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def missing_axes(self, spec) -> list:
        """Required axes with no load case in the specification."""
        stated = {lc.direction for lc in spec.load_cases}
        return [ax for ax in self.required_load_axes if ax not in stated]

    def summary_for_prompt(self) -> str:
        lines = ["DECLARED DESIGN INTENT (an input from the engineer, not a computed result):"]
        lines.append(f"  load axes the part must carry: {', '.join(self.required_load_axes)}")
        for ax in self.required_load_axes:
            why = self.axis_rationale.get(ax) or AXIS_MEANING.get(ax, "")
            lines.append(f"    {ax}: {why}")
        if self.fixed_geometry:
            lines.append(f"  geometry the engineer states is FIXED: {', '.join(self.fixed_geometry)}")
        if self.mutable_geometry:
            lines.append(f"  geometry the engineer permits changing: {', '.join(self.mutable_geometry)}")
        lines.append(f"  stated stress allowable is {'FIRM' if self.stress_allowable_is_firm else 'negotiable'}")
        lines.append(f"  stated mass budget is {'FIRM' if self.mass_budget_is_firm else 'negotiable'}")
        if self.notes:
            lines.append(f"  notes: {self.notes}")
        lines.append(f"  provenance: {self.provenance}")
        return "\n".join(lines)

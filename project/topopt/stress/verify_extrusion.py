"""Physical verification of the 2.5D extruded model.

Two questions the gradient check cannot answer:

1. **Does the extrusion agree with the plane-stress model in-plane?**  It should
   not agree exactly -- a single-layer solid constrains the thickness direction
   through the Poisson effect, so it sits between plane stress and plane strain --
   but the difference must be small and must not grow with layer count.
2. **Is the transverse response trustworthy?**  Trilinear hexes with full
   integration shear-lock in bending when few elements span the thickness, so the
   model is too stiff and *underestimates* transverse displacement and stress.
   That error is in the non-conservative direction for detecting overstress, so any
   transverse finding must be reported with its layer count and refinement trend.

Both are answered on the *solid* structure (density 1 everywhere), which isolates
the finite element model from the optimizer.

Run:  python -m project.topopt.stress.verify_extrusion
"""
from __future__ import annotations

import json

import numpy as np

from project.topopt.stress.problem import Geometry, LoadCase, Optimizer, StressSpec
from project.topopt.stress.solver import Evaluator


def _solid_state(*, model: str, n_layers: int, thickness: float, direction: str,
                 magnitude: float, n_cells: int = 40) -> dict:
    spec = StressSpec(
        formulation="P3",
        geometry=Geometry(n_cells_per_side=n_cells, model=model, n_layers=n_layers,
                          thickness=thickness, exclude_load_elements=True),
        optimizer=Optimizer(max_iter=1, r0_elements=1.5),
    )
    spec.load_cases = [LoadCase(magnitude=magnitude, direction=direction, name=direction)]
    ev = Evaluator(spec)
    st = ev.state(np.ones(ev.n_inplane))
    c = st["cases"][0]
    axis = {"x": 0, "y": 1, "z": 2}[direction]
    ndof = 2 if model == "plane_stress" else 3
    tip = float(c["u"][np.argmax(np.abs(c["u"]))]) if ndof == 3 else 0.0
    return {"model": model, "n_layers": n_layers, "thickness": thickness,
            "direction": direction, "n_elem": int(ev.n_elem),
            "n_dof": int(ev.mesh.n_dof), "compliance": c["compliance"],
            "peak_vM": float(c["svm"].max()), "mass_g": st["mass"] * 1e6,
            "max_abs_disp": abs(tip) if ndof == 3 else float(np.abs(c["u"]).max())}


def beam_out_of_plane(thickness: float, *, magnitude: float, arm_len: float = 120.0,
                      width: float = 80.0, E: float = 71000.0) -> dict:
    """Euler-Bernoulli estimate for out-of-plane bending of the horizontal arm.

    A lower bound on deflection: it ignores the vertical arm's compliance and any
    twist, so the true structure is softer than this.
    """
    I = width * thickness ** 3 / 12.0
    delta = magnitude * arm_len ** 3 / (3.0 * E * I)
    section_modulus = I / (thickness / 2.0)
    sigma = magnitude * arm_len / section_modulus
    return {"delta_mm": delta, "sigma_MPa": sigma, "I_mm4": I}


def run() -> dict:
    inplane = [_solid_state(model="plane_stress", n_layers=1, thickness=1.0,
                            direction="y", magnitude=1500.0)]
    for nl in (1, 2, 4):
        inplane.append(_solid_state(model="extruded_3d", n_layers=nl, thickness=1.0,
                                    direction="y", magnitude=1500.0))

    transverse = []
    for th in (1.0, 20.0):
        for nl in (1, 2, 4, 8):
            transverse.append(_solid_state(model="extruded_3d", n_layers=nl, thickness=th,
                                           direction="z", magnitude=150.0))

    print("=== in-plane agreement, solid structure, 1500 N ===")
    ref = inplane[0]
    for r in inplane:
        tag = "plane stress" if r["model"] == "plane_stress" else f"extruded, {r['n_layers']} layer(s)"
        d = 100.0 * (r["compliance"] / ref["compliance"] - 1.0)
        print(f"  {tag:26s} C = {r['compliance']:9.2f}  peak vM {r['peak_vM']:8.1f} MPa"
              f"  mass {r['mass_g']:6.2f} g   dC vs plane stress {d:+6.2f}%")

    print("\n=== transverse response, solid structure, 150 N out of plane ===")
    for th in (1.0, 20.0):
        est = beam_out_of_plane(th, magnitude=150.0)
        print(f"  thickness {th:5.1f} mm  (beam estimate: delta {est['delta_mm']:12.3f} mm,"
              f" sigma {est['sigma_MPa']:10.1f} MPa)")
        for r in [x for x in transverse if x["thickness"] == th]:
            print(f"    {r['n_layers']:>2} layer(s): |u|max {r['max_abs_disp']:12.4f} mm"
                  f"  C {r['compliance']:12.3e}  peak vM {r['peak_vM']:10.1f} MPa"
                  f"  ({r['n_dof']:>6} dof)")

    payload = {"in_plane": inplane, "transverse": transverse,
               "beam_estimates": {str(th): beam_out_of_plane(th, magnitude=150.0)
                                  for th in (1.0, 20.0)}}
    return payload


if __name__ == "__main__":
    out = run()
    print("\n" + json.dumps({"n_configurations": len(out["in_plane"]) + len(out["transverse"])}))

"""Suggested problems for the New problem page: the benchmark as an editable brief, then walkthrough examples."""
from project.solver.lbracket2d.config import reference_config

_COUNTS = dict(enumerate(("No", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten")))


def _benchmark_brief() -> dict:
    """The simplified_2D_holes benchmark's documented inputs (reference_config) as a design brief in the outline
    format. Started as a new problem, it goes through the parse, the checks and the review like any brief, so the
    engineer can change anything first. Worded as the walkthrough, whose wording the parser records without adding
    assumptions: isotropy in the material model, no method in the purpose."""
    c = reference_config()
    mm = lambda metres: f"{metres * 1e3:g}"
    holes = [f"({mm(x)}, {mm(y)})" for x, y in c.hole_centers_m]
    fx, fy = c.force_n
    if fx == 0 or fy == 0:
        force = f"{abs(fx or fy) / 1e3:g} kN in the {'-' if (fx or fy) < 0 else '+'}{'y' if fx == 0 else 'x'} direction"
    else:
        force = f"({fx:g}, {fy:g}) N, as x and y components"
    stress = (f"Constrain the calibrated volume-averaged p = {c.p_norm:g} stress norm over the L domain to at most "
              f"{c.stress_limit_pa / 1e6:g} MPa." if c.stress_limit_pa else "No stress constraint.")
    request = (
        "Part and purpose: the simplified_2D_holes benchmark, an equal-arm L-bracket plate designed for minimum "
        "compliance.\n\n"
        f"Design space: a {mm(c.lx_m)} by {mm(c.ly_m)} mm outer square with a {mm(c.cut_length_m)} by "
        f"{mm(c.cut_length_m)} mm upper-right square cut, so each arm is {mm(c.lx_m - c.cut_length_m)} mm wide; "
        f"{mm(c.thickness_m)} mm thick, analysed as a flat plate in plane stress. The re-entrant corner is sharp. "
        f"{_COUNTS[len(holes)]} initial through-holes of radius {mm(c.hole_radius_m)} mm at {', '.join(holes[:-1])} and "
        f"{holes[-1]} mm; they may move, merge or close.\n\n"
        f"Material and data source: uniform isotropic material model, E = {c.youngs_modulus_pa / 1e9:g} GPa, "
        f"nu = {c.poisson_ratio:g}; static small-strain linear elasticity.\n\n"
        "Supports: the whole top edge of the vertical arm is clamped to a frame: zero displacement in x and y "
        "along that edge.\n\n"
        f"Loads: a total force of {force}, spread uniformly over a {mm(c.load_patch_m)} mm high band at the upper "
        "end of the horizontal-arm tip edge, through the full thickness.\n\n"
        f"Objective and limits: minimize compliance; keep at most {c.volume_fraction:.0%} material, measured "
        f"against the full L-shaped area excluding the corner cut, before subtracting the initial holes. {stress}\n\n"
        "Manufacturing: none."
    )
    context = ("Coordinates: x right, y up, origin at the lower-left corner of the outer square. These are the "
               "documented inputs of the mfrto simplified_2D_holes benchmark"
               + (f". The {c.stress_limit_pa / 1e6:g} MPa value is a calibrated stress-norm limit for "
                  f"p = {c.p_norm:g}, not a yield or peak-stress bound." if c.stress_limit_pa else "."))
    return {"request": request, "context": context}


BENCHMARK_EXAMPLE = "Benchmark: 2D L-bracket plate with five initial holes (simplified_2D_holes)"

EXAMPLES = {
    BENCHMARK_EXAMPLE: _benchmark_brief(),
    "2D L-bracket: walkthrough (missing support, load band, volume reference)": {
        "request": (
            "Part and purpose: a titanium L-bracket plate; minimum-compliance level-set design.\n\n"
            "Design space: 100 by 100 mm outer square with a 60 by 60 mm upper-right cut, 12 mm "
            "thick, analysed as a flat plate in plane stress. Five initial holes of radius 10 mm at "
            "(20,20), (20,50), (20,80), (50,20) and (80,20) mm; they may move, merge or close.\n\n"
            "Material and data source: uniform isotropic titanium model, E = 120 GPa, nu = 0.36; "
            "static small-strain linear elasticity.\n\n"
            "Supports: the vertical arm is bolted to a frame at its top.\n\n"
            "Loads: a 5 kN downward load at the tip of the horizontal arm.\n\n"
            "Objective and limits: minimize compliance; keep at most 75% material. Include the "
            "calibrated volume-averaged p = 6 stress norm over the L domain, limit 125 MPa.\n\n"
            "Manufacturing: none."
        ),
        "context": (
            "Coordinates: x right, y up, origin at the lower-left corner. The 125 MPa value is a "
            "calibrated stress-norm limit, not a yield or peak-stress bound."
        ),
        # Prepared engineer answers for a walkthrough. solver_* keys match code-check question keys
        # and can be filled automatically; "reviewer (...)" entries are shown for copying, because
        # reviewer question keys vary.
        "answers": {
            "solver_support": "The whole top edge of the vertical arm is clamped to the frame: "
                              "zero displacement in x and y along that edge.",
            "solver_dimensions": "Spread the 5 kN total force uniformly over a 4 mm high band at the upper end "
                                 "of the horizontal-arm tip edge, through the full thickness, acting in -y.",
            "solver_volume_reference": "75% of the full L-shaped area, excluding the corner cut, "
                                       "before subtracting the initial holes.",
            "reviewer (stress basis)": "The 125 MPa limit is calibrated for this benchmark with p = 6.",
            "reviewer (corner fillet)": "No fillet; keep the sharp re-entrant corner as in the benchmark.",
        },
    },
    "Unsupported shape: FRC intake pivot side plate": {
        "request": (
            "Design a flat 10 in by 6 in, 1/4 in thick side plate for the robot intake. "
            "Attach the plate to the robot using the existing mounting holes and retain "
            "the current intake pivot location. Half pocket this plate. Apply a 125 lbf "
            "load into the flat plate side to represent an impact from another robot. "
            "Also include a 125 lbf load into the front side. Minimize compliance using "
            "75% of the available material. Keep the mounting-hole and pivot regions solid. "
            "The part will be cut from 0.25 in polycarbonate on the team's CNC router."
        ),
        "context": (
            "The front side is the left-most edge face when viewing the flat side of the plate. "
            "Use y upward, x normal to the flat face, and z parallel to the plane of the plate. "
            "The intake pivot location must remain unchanged. A bumper occupies nearby clearance "
            "that the plate must be designed around. The front-side load is distributed across "
            "the face. The side-impact load is also intended as a distributed face load."
        ),
    }
}

EXAMPLES = {
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
    "2D L-bracket: explicit five-hole benchmark": {
        "request": (
            "Use the simplified_2D_holes benchmark: an equal-arm 100 mm by 100 mm L plate, "
            "12 mm thick, plane stress, with a 60 mm upper-right square cut. Use five movable "
            "initial through-holes, radius 10 mm, at (20,20), (20,50), (20,80), (50,20), (80,20) mm. "
            "The holes may move, merge or close. Use static small-strain linear isotropic "
            "elasticity, uniform E=120 GPa and nu=0.36. Clamp the full top edge of the vertical arm "
            "in x and y. Distribute a total force [0,-5000] N over the 4 mm high band at the upper "
            "end of the horizontal-arm tip edge, through the full thickness. Minimize compliance "
            "with at most 75% material relative to the full L domain excluding the square cut. "
            "Also use the calibrated volume-averaged p=6 stress norm constraint, upper limit "
            "125 MPa over the L domain. No manufacturing constraints."
        ),
        "context": (
            "Coordinates: x right, y up, origin at lower-left. This is the mfrto simplified_2D_holes "
            "case. The 125 MPa norm limit is calibrated; it is not a material yield limit or peak "
            "stress bound."
        ),
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

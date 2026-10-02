EXAMPLES = {
    "3D L-bracket: start with questions": {
        "request": "I want to optimize a 3D L-bracket. Help me specify the bracket, its loading and supports before running it.",
        "context": "Ask about unresolved physical decisions. Do not fill in benchmark dimensions, material properties, loads or constraints for me.",
    },
    "3D L-bracket: explicit five-hole benchmark": {
        "request": (
            "Use the simplified_3D_holes benchmark: an equal-arm 100 mm by 100 mm L, "
            "12 mm thick, with a 60 mm upper-right square cut through thickness. "
            "Use five movable initial through-holes, radius 10 mm, at (20,20), (20,50), "
            "(20,80), (50,20), (80,20) mm. The holes may move, merge or close. "
            "Use static small-strain linear isotropic 3D solid elasticity, uniform "
            "E=120 GPa and nu=0.36. Clamp the full top face of the vertical arm in x,y,z. "
            "Distribute a total force [0,-5000,0] N over the 4 mm high band at the upper "
            "end of the horizontal-arm tip face, through the full thickness. Minimize "
            "compliance with at most 75% material relative to the full L domain excluding "
            "the square cut. Also use the calibrated volume-averaged p=6 stress norm "
            "constraint, upper limit 116 MPa over the L domain. No manufacturing constraints."
        ),
        "context": (
            "Coordinates: x right, y up, z through thickness, origin at lower-left. "
            "This explicitly selected reference is a 3D extension inspired by Fig.5, "
            "not a reproduction of the paper. The 116 MPa norm limit is calibrated; "
            "it is not a material yield limit or peak stress bound."
        ),
    },
    "M2: FRC intake pivot side plate": {
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

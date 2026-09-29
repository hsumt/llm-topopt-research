EXAMPLES = {
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
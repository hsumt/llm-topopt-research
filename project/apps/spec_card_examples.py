"""Three pilot specification-card prompts for quick Streamlit smoke testing."""

SPEC_CARD_EXAMPLES = {
    "M2 — FRC intake pivot side plate": {
        "problem": (
            "Design a flat 10 in by 6 in, ¼ in thick side plate for the robot intake. "
            "Attach the plate to the robot using the existing mounting holes and retain "
            "the current intake pivot location. Half pocket this plate. Apply a 125lb load "
            "into the flat plate side to represent the impact load of being hit from the side "
            "by another robot. Also include a 125 lb load into the front side. Minimize "
            "compliance using 75% of the available material. Keep the mounting-hole and pivot "
            "regions solid. The part will be cut from 0.25 in polycarbonate on the team's CNC router."
        ),
        "context": (
            "The front side is the side on the left of the flat face; viewed from the flat "
            "side of the plate, it is the left-most edge face. Use y upward, x normal to the "
            "flat face, and z parallel to the plane of the flat face. The intake pivot location "
            "must remain unchanged. A bumper occupies nearby clearance that the plate must be "
            "designed around; in the existing sketch the pivot is marked with a red X and the "
            "bumper cross-section is blue when looking down the x-axis. The front-side load is "
            "distributed across the face. The side-impact load is also intended as a distributed "
            "face load. Cross-sectional manufacturing error is at most 0.001 in; thickness error "
            "is considered negligible for this pilot."
        ),
    },
    "M1 — FRC rotating shooter turret base plate": {
        "problem": (
            "Design a flat 10 in by 10 in, ¼ inch thick base plate for the rotating shooter "
            "turret. Attach the plate to the existing robot superstructure at the back mounting "
            "boxtubes and retain the existing turret rotation interface. Apply a load over the "
            "turret mounting region to represent the weight of the 15-lb shooter assembly. "
            "Minimize compliance using 35% of the available material. The part will be cut from "
            "6061 Aluminum using the team's CNC router. Additionally, the turret plate material "
            "must be rounded around the area where the shooter sits upon."
        ),
        "context": (
            "Existing back mounting boxtubes and the current turret rotation interface are fixed "
            "interfaces from the current robot. The shooter assembly weighs approximately 15 lb. "
            "The rounding requirement is a physical packaging/manufacturing requirement around "
            "the shooter seating region, but its exact radius/clearance geometry has not yet been stated."
        ),
    },
    "M3 — Pipetting-machine Z-axis carriage plate": {
        "problem": (
            "Design a flat 60 mm by 60 mm, 10 mm thick carriage plate for the Z-axis of the "
            "automated pipetting machine. Mount the carriage to the existing vertical linear-motion "
            "assembly using the current mounting locations. The pipetting mechanism applies an "
            "approximately 2 N downward load at its mounting region on the carriage. Minimize "
            "compliance using 45% of the available material. Keep the existing mounting regions "
            "solid. The part will be 3D printed from PETG-CF."
        ),
        "context": (
            "The current vertical linear-motion assembly and its mounting locations already exist "
            "and should be retained. The carriage plate will be additively manufactured from "
            "PETG-CF; print orientation and detailed process constraints are not yet specified."
        ),
    },
}

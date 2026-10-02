from project.models import ProblemSpec


def problem_graph(spec: ProblemSpec) -> str:
    lines = [
        "digraph problem {",
        "rankdir=LR;",
        'node [shape=box, style="rounded"];',
        f'problem [label="{spec.name}\\n{spec.problem_kind}"];',
    ]

    if spec.geometry:
        label = spec.geometry.description or "Geometry"
        lines.append(f'geometry [label="Geometry\\n{_clean(label)}"];')
        lines.append("problem -> geometry;")

    if spec.physics:
        lines.append(
            f'physics [label="Physics\\n{_clean(spec.physics.family)}"];'
        )
        lines.append("problem -> physics;")

    if spec.materials:
        names = ", ".join(
            material.name or "unspecified"
            for material in spec.materials
        )
        lines.append(
            f'materials [label="Material\\n{_clean(names)}"];'
        )
        lines.append("problem -> materials;")

    if spec.boundary_conditions:
        names = ", ".join(
            condition.name
            for condition in spec.boundary_conditions
        )
        lines.append(
            f'supports [label="Supports\\n{_clean(names)}"];'
        )
        lines.append("problem -> supports;")

    if spec.loads:
        names = ", ".join(
            load.name
            for load in spec.loads
        )
        lines.append(
            f'loads [label="Loads\\n{_clean(names)}"];'
        )
        lines.append("problem -> loads;")

    if spec.optimization:
        objectives = ", ".join(
            objective.quantity or "unspecified"
            for objective in spec.optimization.objectives
        )
        lines.append(
            f'optimization [label="Optimization\\n{_clean(objectives)}"];'
        )
        lines.append("problem -> optimization;")

    if spec.manufacturing:
        label = spec.manufacturing.process or "Manufacturing"
        lines.append(
            f'manufacturing [label="Manufacturing\\n{_clean(label)}"];'
        )
        lines.append("problem -> manufacturing;")

    lines.append("}")
    return "\n".join(lines)
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from project.models import ProblemSpec


def geometry_preview(spec: ProblemSpec) -> Figure | None:
    if spec.geometry is None:
        return None

    parameters = spec.geometry.parameters

    length = _number(parameters, "length")
    height = _number(parameters, "height", "height_y", "width_z", "width")
    thickness = _number(parameters, "thickness", "thickness_x")

    if length is None or height is None:
        return None

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.add_patch(
        plt.Rectangle(
            (0, 0),
            length,
            height,
            fill=False,
            linewidth=2,
        )
    )

    ax.set_xlim(-0.08 * length, 1.08 * length)
    ax.set_ylim(-0.15 * height, 1.15 * height)
    ax.set_aspect("equal")

    ax.set_xlabel("z / plate length")
    ax.set_ylabel("y / plate height")

    title = f"Physical geometry preview — {length:g} × {height:g}"

    if thickness is not None:
        title += f" × {thickness:g}"

    ax.set_title(title)

    unresolved = []

    for region in spec.geometry.regions:
        name = region.name.lower()

        if (
            "mount" in name
            or "pivot" in name
            or "bumper" in name
        ):
            unresolved.append(region.name)

    # if unresolved:
    #     ax.text(
    #         length / 2,
    #         -0.10 * height,
    #         "CAD-governed regions not positioned in this preview:\n"
    #         + ", ".join(unresolved),
    #         ha="center",
    #         va="top",
    #         fontsize=9,
    #     )

    ax.text(
        length / 2,
        height / 2,
        "Design plate",
        ha="center",
        va="center",
    )

    ax.grid(False)

    return fig


def _number(
    parameters,
    *names: str,
) -> float | None:
    for name in names:
        item = parameters.get(name)

        if item is None:
            continue

        try:
            return float(item.value)
        except (TypeError, ValueError):
            continue

    return None


def _clean(text: str) -> str:
    return (
        str(text)
        .replace('"', "'")
        .replace("\n", " ")
    )
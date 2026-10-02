import math

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon

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
def geometry_preview(spec: ProblemSpec) -> Figure | None:
    if spec.geometry is None:
        return None

    if spec.geometry.template == "lbracket3d_five_holes":
        return _lbracket_preview(spec)

    parameters = spec.geometry.parameters

    length = _number(parameters, "length", "width_z", "width")
    height = _number(parameters, "height", "height_y")
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


_LENGTH_TO_MM = {"m": 1000., "cm": 10., "mm": 1., "in": 25.4,
                 "inch": 25.4, "inches": 25.4}
_FORCE_TO_N = {"n": 1., "kn": 1000., "lbf": 4.4482216152605}


def _length_mm(parameters, name: str) -> float | None:
    item = parameters.get(name)
    if item is None or isinstance(item.value, bool):
        return None
    factor = _LENGTH_TO_MM.get((item.unit or "").strip().lower())
    if factor is None:
        return None
    try:
        number = float(item.value) * factor
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _force_vector(load) -> tuple[float, float, float] | None:
    """Draw only loads whose location, units and direction are explicit."""
    if load.location != "tip_band" or load.kind != "total_force" or load.magnitude is None:
        return None
    factor = _FORCE_TO_N.get((load.magnitude.unit or "").strip().lower())
    if factor is None:
        return None
    raw = load.magnitude.value
    try:
        if isinstance(raw, (list, tuple)):
            if len(raw) != 3 or load.direction not in {None, "vector"} or any(isinstance(v, bool) for v in raw):
                return None
            vector = tuple(float(v) * factor for v in raw)
        else:
            directions = {"x": (1, 0, 0), "+x": (1, 0, 0), "-x": (-1, 0, 0),
                          "y": (0, 1, 0), "+y": (0, 1, 0), "-y": (0, -1, 0),
                          "z": (0, 0, 1), "+z": (0, 0, 1), "-z": (0, 0, -1)}
            direction = directions.get(load.direction)
            if direction is None or isinstance(raw, bool) or float(raw) <= 0:
                return None
            vector = tuple(float(raw) * factor * value for value in direction)
    except (ValueError, TypeError):
        return None
    return vector if all(math.isfinite(v) for v in vector) and any(vector) else None


def _lbracket_preview(spec: ProblemSpec) -> Figure | None:
    geometry = spec.geometry
    parameters = geometry.parameters
    lx = _length_mm(parameters, "lx")
    ly = _length_mm(parameters, "ly")
    cut = _length_mm(parameters, "cut_length")
    if lx is None or ly is None or cut is None or cut >= min(lx, ly):
        return None
    thickness = _length_mm(parameters, "thickness")
    patch = _length_mm(parameters, "load_patch")
    radius = _length_mm(parameters, "hole_radius")
    arm_x, arm_y = lx - cut, ly - cut
    fig, ax = plt.subplots(figsize=(7, 5.7))
    outline = Polygon([(0, 0), (lx, 0), (lx, arm_y), (arm_x, arm_y),
                       (arm_x, ly), (0, ly)], closed=True,
                      facecolor="#dbe9f1", edgecolor="#294c60", linewidth=1.8,
                      label="L-shaped domain")
    ax.add_patch(outline)
    ax.text(arm_x + cut / 2, arm_y + cut / 2,
            f"Square cut\n{cut:g} × {cut:g} mm", ha="center", va="center",
            color="#64748b", fontsize=10)

    centers = parameters.get("hole_centers")
    if centers is not None and radius is not None:
        factor = _LENGTH_TO_MM.get((centers.unit or "").strip().lower())
        initial = geometry.hole_role == "initial_void"
        hole_label = "Initial holes (may change)" if initial else "Specified holes"
        if factor is not None and isinstance(centers.value, (list, tuple)):
            labeled = False
            for center in centers.value:
                try:
                    if len(center) != 2 or any(isinstance(v, bool) for v in center):
                        continue
                    x, y = (float(v) * factor for v in center)
                    if not all(math.isfinite(v) for v in (x, y)):
                        continue
                except (TypeError, ValueError):
                    continue
                circle = Circle((x, y), radius, facecolor="white", edgecolor="#42677e",
                                linewidth=1.2, linestyle="--" if initial else "-",
                                label=hole_label if not labeled else None)
                circle.set_clip_path(outline)
                ax.add_patch(circle)
                labeled = True

    clamped = any(bc.location == "top_arm" and (bc.kind or "").lower() in {"fixed", "clamped"}
                  and sorted(bc.components) == ["x", "y", "z"]
                  and (bc.value is None or bc.value.value == 0)
                  for bc in spec.boundary_conditions)
    if clamped:
        ax.plot([0, arm_x], [ly, ly], linewidth=3, color="#334155", label="Top face clamped")
        ax.add_patch(plt.Rectangle((0, ly), arm_x, .025 * ly, facecolor="none",
                                  edgecolor="#64748b", hatch="////", linewidth=.6))

    load_notes = []
    arrow_ends = []
    if patch is not None and patch <= arm_y:
        for load in spec.loads:
            force = _force_vector(load)
            if force is None:
                continue
            fx, fy, fz = force
            y = arm_y - patch / 2
            ax.plot([lx, lx], [arm_y - patch, arm_y], color="#c2410c", linewidth=4)
            in_plane = math.hypot(fx, fy)
            if in_plane:
                scale = .20 * max(lx, ly) / in_plane
                endpoint = (lx + fx * scale, y + fy * scale)
                arrow_ends.append(endpoint)
                ax.annotate("", xy=endpoint, xytext=(lx, y),
                            arrowprops={"arrowstyle": "-|>", "lw": 2, "color": "#c2410c"})
                load_notes.append(f"Fx = {fx:g} N; Fy = {fy:g} N")
            if fz:
                ax.text(lx, y, r"$\odot$" if fz > 0 else r"$\otimes$",
                        ha="center", va="center", color="#c2410c", fontsize=19)
                load_notes.append(f"Fz = {fz:+g} N ({'out of' if fz > 0 else 'into'} plane)")
    if load_notes:
        ax.text(.99, .01, "\n".join(load_notes), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, color="#9a3412")

    title = f"L-bracket • {lx:g} × {ly:g} mm"
    title += f" • thickness {thickness:g} mm" if thickness is not None else " • thickness unspecified"
    ax.set_title(title, fontsize=11, pad=12)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_xlim(-.08 * lx, max([1.30 * lx] + [x + .06 * lx for x, _ in arrow_ends]))
    ax.set_ylim(min([-.16 * ly] + [y - .07 * ly for _, y in arrow_ends]),
                max([1.12 * ly] + [y + .07 * ly for _, y in arrow_ends]))
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.text(.5, .01, "x–y footprint; thickness extends along z. Dimensions and loads shown only when specified.",
             ha="center", fontsize=8, color="#64748b")
    fig.tight_layout(rect=(0, .045, 1, 1))
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

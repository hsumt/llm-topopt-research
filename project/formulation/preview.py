"""Deterministic intent preview generated directly from ``ProblemSpec``.

The preview is not an LLM rendering. It is drawn from the exact structured
values that will be passed to the deterministic solver so the visualization
cannot silently disagree with the runnable specification.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


_POINT_COORDS = {
    "bottom_left": (0.0, 0.0),
    "bottom_right": (1.0, 0.0),
    "top_left": (0.0, 1.0),
    "top_right": (1.0, 1.0),
    "right_tip": (1.0, 0.5),
    "right_center": (1.0, 0.5),
    "left_center": (0.0, 0.5),
    "top_center": (0.5, 1.0),
    "bottom_center": (0.5, 0.0),
}


def _location_xy(location: str, Lx: float, Ly: float) -> tuple[float, float]:
    if location in _POINT_COORDS:
        fx, fy = _POINT_COORDS[location]
        return fx * Lx, fy * Ly
    centers = {
        "left_edge": (0.0, Ly / 2.0),
        "right_edge": (Lx, Ly / 2.0),
        "top_edge": (Lx / 2.0, Ly),
        "bottom_edge": (Lx / 2.0, 0.0),
    }
    if location in centers:
        return centers[location]
    raise ValueError(f"Unsupported preview location: {location}")


def _load_label(load) -> str:
    component = "F" + load.dof
    if load.kind == "point_force":
        return f"point force: {component}={load.value:g}"
    if load.kind == "edge_resultant":
        return f"edge resultant: total {component}={load.value:g}"
    if load.kind == "edge_traction":
        return f"edge traction: q{load.dof}={load.value:g} per unit length"
    return f"{load.kind}: {load.dof}={load.value:g}"


def _draw_load(ax, load, Lx: float, Ly: float) -> None:
    scale = 0.16 * max(Lx, Ly)
    dof = load.dof
    sign = 1.0 if load.value >= 0 else -1.0

    if load.kind == "point_force":
        x, y = _location_xy(load.location, Lx, Ly)
        if dof == "y":
            dx, dy = 0.0, sign * scale
        else:
            dx, dy = sign * scale, 0.0
        ax.annotate(
            "",
            xy=(x, y),
            xytext=(x - dx, y - dy),
            arrowprops={"arrowstyle": "->", "linewidth": 2.1},
        )
        ax.text(
            x,
            y + 0.04 * Ly,
            _load_label(load),
            fontsize=8,
            ha="center",
            va="bottom",
        )
        return

    n_arrows = 6
    fractions = [(i + 1) / (n_arrows + 1) for i in range(n_arrows)]
    if load.location in {"left_edge", "right_edge"}:
        x = 0.0 if load.location == "left_edge" else Lx
        points = [(x, f * Ly) for f in fractions]
    elif load.location in {"bottom_edge", "top_edge"}:
        y = 0.0 if load.location == "bottom_edge" else Ly
        points = [(f * Lx, y) for f in fractions]
    else:
        raise ValueError(
            f"{load.kind} requires an edge location for deterministic preview"
        )

    for x, y in points:
        if dof == "y":
            dx, dy = 0.0, sign * scale
        else:
            dx, dy = sign * scale, 0.0
        ax.annotate(
            "",
            xy=(x, y),
            xytext=(x - dx, y - dy),
            arrowprops={"arrowstyle": "->", "linewidth": 1.2},
        )
    cx, cy = _location_xy(load.location, Lx, Ly)
    ax.text(
        cx,
        cy + 0.07 * Ly,
        _load_label(load),
        fontsize=8,
        ha="center",
        va="bottom",
    )


def format_intent_snapshot(spec) -> str:
    """Compact terminal summary of the same values rendered in the PNG."""
    Lx = float(spec.mesh.Lx)
    Ly = float(spec.mesh.Ly)
    lines = [
        f"  Domain: {Lx:g} x {Ly:g} ({Lx / Ly:g}:1 length:height)",
        f"  Analysis: {spec.analysis.formulation}, {spec.analysis.unit_system}",
    ]

    grouped: dict[str, list[str]] = defaultdict(list)
    for bc in spec.bcs:
        grouped[bc.location].append(f"u{bc.dof}={bc.value:g}")
    for location, labels in grouped.items():
        lines.append(f"  Support: {location} -> {', '.join(labels)}")

    for load in spec.loads:
        lines.append(
            f"  Load: {load.location} -> {_load_label(load)}"
        )

    lines.append(
        f"  Objective: compliance minimization; volume fraction={spec.simp.vol_frac:g}"
    )
    return "\n".join(lines)


def generate_intent_preview(
    spec,
    output_path: str | Path,
    *,
    title: str = "Deterministic pre-solve intent preview",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    Lx = float(spec.mesh.Lx)
    Ly = float(spec.mesh.Ly)

    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.add_patch(Rectangle((0.0, 0.0), Lx, Ly, fill=False, linewidth=2.0))

    bcs_by_location: dict[str, list[str]] = defaultdict(list)
    for bc in spec.bcs:
        bcs_by_location[bc.location].append(f"u{bc.dof}={bc.value:g}")

    for location, labels in bcs_by_location.items():
        x, y = _location_xy(location, Lx, Ly)
        if location == "left_edge":
            ax.plot([0.0, 0.0], [0.0, Ly], linewidth=4.0)
        elif location == "right_edge":
            ax.plot([Lx, Lx], [0.0, Ly], linewidth=4.0)
        elif location == "bottom_edge":
            ax.plot([0.0, Lx], [0.0, 0.0], linewidth=4.0)
        elif location == "top_edge":
            ax.plot([0.0, Lx], [Ly, Ly], linewidth=4.0)
        else:
            ax.plot([x], [y], marker="s", markersize=7)

        offset_x = -0.04 * Lx if "left" in location else 0.04 * Lx
        ha = "right" if "left" in location else "left"
        if location in {"top_edge", "bottom_edge", "top_center", "bottom_center"}:
            offset_x = 0.0
            ha = "center"
        ax.text(
            x + offset_x,
            y,
            f"BC: {location}\n" + ", ".join(labels),
            fontsize=8,
            ha=ha,
            va="center",
        )

    for load in spec.loads:
        _draw_load(ax, load, Lx, Ly)

    ax.set_xlim(-0.28 * Lx, 1.28 * Lx)
    ax.set_ylim(-0.22 * Ly, 1.28 * Ly)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    summary = (
        f"Domain: {Lx:g} x {Ly:g} | Analysis: {spec.analysis.formulation} | "
        f"E={spec.material.E:g}, nu={spec.material.nu:g}\n"
        f"Objective: compliance minimization | Volume fraction: {spec.simp.vol_frac:g} | "
        f"Unit system: {spec.analysis.unit_system}"
    )
    fig.text(0.5, 0.02, summary, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.98))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path

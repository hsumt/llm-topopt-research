"""Figures for an iterative cascade run.

Two deliverables:

``solutions_figure``
    One column per accepted design state, from the stated problem through each
    authorised edit set. Density on the top row, recovered von Mises on the bottom,
    with the per-load-case peak against the allowable printed on each panel. The
    point of the figure is that the design states are comparable: same domain, same
    colour scale, same allowable.

``hypothesis_graph``
    The state of the hypothesis space after the run. Singleton hypotheses across the
    top, every enumerated subset below, an edge from each singleton to the subsets
    containing it, and nodes coloured by the verdict the re-solve returned. The
    subset layer is drawn in full rather than filtered, because the reason it exists
    is that a hypothesis eliminated alone can re-enter in combination -- and that is
    only visible if the eliminated singletons are on the page next to the surviving
    pair.
"""
from __future__ import annotations

import numpy as np

VERDICT_COLOUR = {
    "SURVIVES": "#2e7d32",
    "ELIMINATED": "#b0b7bd",
    "INFEASIBLE": "#e07b39",
    "NOT_REPRESENTABLE": "#3f7cac",
    "NOT_ASSESSED": "#d9d2c5",
}
VERDICT_MEANING = {
    "SURVIVES": "re-solved; anomaly entailed away",
    "ELIMINATED": "re-solved; anomaly persists",
    "INFEASIBLE": "no feasible design exists",
    "NOT_REPRESENTABLE": "model cannot be asked",
    "NOT_ASSESSED": "no trustworthy verdict",
}
TIER_MARK = {"T1": "\u25cb", "T2": "\u25d1", "T3": "\u25cf"}

#: Filler tokens carried by long operation names, especially synthesized ones.
_FILLER = {"set", "add", "to", "the", "of", "a", "an", "for", "case", "switch",
           "declare", "enable", "raise", "make", "with", "and", "model", "variable",
           "resolution", "value", "new", "on", "in", "as", "per", "its", "into"}
_ABBREV = (
    ("out_of_plane", "z load case"), ("outofplane", "z load case"),
    ("transverse", "z load case"), ("hand_load", "z load case"),
    ("extruded_3d", "2.5D model"), ("extruded", "2.5D model"),
    ("stress_limit", "relax allowable"), ("mass_budget", "raise mass"),
    ("mass_fraction", "raise mass"), ("distribute", "spread load"),
    ("corner_fillet", "corner fillet"), ("fillet", "corner fillet"),
    ("refine_mesh", "refine mesh"), ("cluster", "stress clusters"),
    ("filter", "filter radius"), ("exclude", "exclude load region"),
    ("stress_constraints", "stress constraints"), ("pnorm", "p-norm exponent"),
)


def short_label(op_name: str, width: int = 20) -> str:
    """Compact node label. Operation names are long, and synthesized ones verbose.

    Abbreviation is by longest-matching pattern first, then by dropping filler
    tokens. Two operations can still collapse onto the same label -- observed with
    ``set_initial_thickness`` and ``set_thickness_to_stock_bound``, which is worse
    than a long label because the figure then shows two different edits as one -- so
    ``unique_labels`` below resolves collisions rather than leaving them.
    """
    low = op_name.lower()
    for key, lab in _ABBREV:
        if key in low:
            return lab
    toks = [t for t in low.split("_") if t and t not in _FILLER]
    lab = " ".join(toks[:3])
    return lab[:width] if len(lab) > width else lab


def unique_labels(op_names) -> dict:
    """Labels guaranteed distinct across the operations in one figure."""
    out, seen = {}, {}
    for name in op_names:
        lab = short_label(name)
        seen.setdefault(lab, []).append(name)
    for lab, names in seen.items():
        if len(names) == 1:
            out[names[0]] = lab
        else:
            for n in names:
                toks = [t for t in n.lower().split("_") if t and t not in _FILLER]
                out[n] = " ".join(toks[:3])[:19] or n[:19]
    return out


def _grid(centroids2d, values, h, n):
    grid = np.full((n, n), np.nan)
    ix = np.round(centroids2d[:, 0] / h - 0.5).astype(int)
    iy = np.round(centroids2d[:, 1] / h - 0.5).astype(int)
    order = np.argsort(values)          # keep the max when layers collapse onto a cell
    grid[iy[order], ix[order]] = values[order]
    return grid


def solutions_figure(states, results, *, path, apply_style=None, panel_letter=None,
                     meta_grey="#5c6672"):
    """Design state per accepted edit set. ``results`` are the RunResult objects."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.patches import Polygon

    if apply_style:
        apply_style()
    ncol = len(results)
    fig, axes = plt.subplots(2, ncol, figsize=(2.35 * ncol + 1.0, 6.3), squeeze=False)
    dens_cmap = LinearSegmentedColormap.from_list("dens", ["#f7f7f7", "#1a1a1a"])

    limit = float(results[0].spec.constraints.stress_limit)
    svmax = max(float(r.sigma_vm.max()) for r in results)
    # Clip the shared colour scale at twice the allowable. A scale stretched to the
    # true maximum makes every other panel unreadable: with one state peaking at 17x
    # the allowable, a 606 MPa field and a 61 MPa field both render as the same pale
    # wash, so the figure would show no difference between a design that is over
    # yield and one that is comfortably under. Clipping keeps the allowable
    # meaningful across panels; the exact peaks are printed under each panel, so
    # nothing is hidden by the saturation.
    cap = 2.0 * limit
    snorm = TwoSlopeNorm(vmin=0.0, vcenter=limit, vmax=cap)
    saturated = svmax > cap

    for col, (st, res) in enumerate(zip(states, results)):
        n = res.spec.geometry.n_cells_per_side
        h = res.spec.geometry.L / n
        L, arm = res.spec.geometry.L, res.spec.geometry.arm_fraction * res.spec.geometry.L
        outline = [(0, 0), (L, 0), (L, arm), (arm, arm), (arm, L), (0, L)]

        ax = axes[0][col]
        im = ax.imshow(_grid(res.centroids2d, res.rho_elem, h, n), origin="lower",
                       extent=[0, L, 0, L], cmap=dens_cmap, vmin=0, vmax=1,
                       interpolation="nearest")
        ax.set_title(f"{col + 1}. {st['label']}", fontsize=7.5)
        if col == ncol - 1:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.set_label("density $\\rho$", fontsize=6)

        ax2 = axes[1][col]
        im2 = ax2.imshow(_grid(res.centroids2d,
                               np.where(res.rho_elem >= 0.5, res.sigma_vm, np.nan), h, n),
                         origin="lower", extent=[0, L, 0, L], cmap="RdYlBu_r",
                         norm=snorm, interpolation="nearest")
        if col == ncol - 1:
            cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03,
                               extend="max" if saturated else "neither")
            cb2.set_label(f"von Mises (MPa)\nwhite = {limit:.0f} allowable"
                          + (f"\nsaturated above {cap:.0f}" if saturated else ""),
                          fontsize=6)

        peaks = st["peak_by_case_MPa"]
        ok = st["within_allowable_by_case"]
        lines = [f"{k}: {v:.0f} MPa  {'within' if ok[k] else 'OVER'} allowable"
                 for k, v in peaks.items()]
        lines.append(f"model: {st['model']}"
                     + (f", {st['n_layers']} layers" if st["model"] != "plane_stress" else "")
                     + f",  t = {st['summary'].get('thickness_mm', res.spec.geometry.thickness):g} mm")
        lines.append(f"mass fraction {st['summary']['mass_fraction']:.3f}")
        fired = st.get("targets_fired") or []
        # Wrap rather than run off the panel: four predicate names on one line
        # overran into the neighbouring column in the first render.
        if fired:
            names = [f.replace("_", " ") for f in fired]
            lines.append("predicates firing:")
            for i in range(0, len(names), 2):
                lines.append("   " + ", ".join(names[i:i + 2]))
        else:
            lines.append("predicates: all quiet")
        ax2.text(0.0, -0.16, "\n".join(lines), transform=ax2.transAxes, fontsize=6.2,
                 va="top", ha="left", color=meta_grey, linespacing=1.45)

        for a in (ax, ax2):
            a.add_patch(Polygon(outline, closed=True, facecolor="none",
                                edgecolor="#5c6672", lw=0.9, zorder=4))
            a.set_xlim(-8, L + 8)
            a.set_ylim(-8, L + 8)
            a.set_xticks([])
            a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)

    axes[0][0].set_ylabel("converged density", fontsize=8)
    axes[1][0].set_ylabel("recovered von Mises", fontsize=8)
    if panel_letter:
        for i, a in enumerate(axes.ravel()):
            panel_letter(a, "abcdefghij"[i])
    fig.suptitle("Accepted design states: each panel is the converged optimum of the "
                 "stated problem after the preceding authorised edits",
                 fontsize=9, x=0.02, y=0.995, ha="left")
    fig.tight_layout(rect=[0, 0.17, 1, 0.965], w_pad=0.6)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def hypothesis_graph(record, *, path, applied_ids=None, apply_style=None,
                     meta_grey="#5c6672", max_subset_nodes=40, tier_map=None):
    """Hypothesis space after one epoch: singletons, subsets, verdicts, couplings."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, FancyArrowPatch

    if apply_style:
        apply_style()
    applied = set(applied_ids or [])
    cands = {c["id"]: c for c in record["candidates"]}
    # Tier belongs to the OPERATION, assigned by the grammar; a candidate dict only
    # carries the tier its proposer suggested, which is exactly the value that must
    # not be displayed. Resolve from the live registry (which includes operations
    # synthesized during the run) and fall back to a caller-supplied map.
    registry = {}
    try:
        from project.ato.grammar import all_operations
        registry = {n: o.tier for n, o in all_operations().items()}
    except Exception:
        pass
    registry.update({n: v.get("tier") for n, v in
                     (record.get("synthesized_operations") or {}).items()})
    registry.update(tier_map or {})
    tiers = {c["id"]: registry.get(c["operation"], "") for c in record["candidates"]}
    labels = unique_labels({c["operation"] for c in record["candidates"]})
    verdict = {tuple(d["ids"]): d["verdict"] for d in record["discharges"]}
    cleared = {tuple(d["ids"]): d.get("cleared", []) for d in record["discharges"]}

    singles = [ids for ids in verdict if len(ids) == 1]
    singles.sort(key=lambda t: t[0])
    subsets = [ids for ids in verdict if len(ids) > 1]
    subsets.sort(key=lambda t: (len(t), t))
    if len(subsets) > max_subset_nodes:
        keep = [s for s in subsets if verdict[s] in ("SURVIVES", "INFEASIBLE",
                                                     "NOT_REPRESENTABLE") or cleared[s]]
        rest = [s for s in subsets if s not in keep]
        subsets = (keep + rest)[:max_subset_nodes]
        subsets.sort(key=lambda t: (len(t), t))

    by_size: dict = {}
    for s in subsets:
        by_size.setdefault(len(s), []).append(s)
    rows = [singles] + [by_size[k] for k in sorted(by_size)]
    n_rows = len(rows)
    width = max(9.0, 1.35 * max(len(r) for r in rows))
    fig, ax = plt.subplots(figsize=(width, 1.95 * n_rows + 1.1))

    pos, rw, rh = {}, 1.08, 0.42
    for ri, row in enumerate(rows):
        y = -ri * 1.6
        span = max(len(row), 1)
        for ci, ids in enumerate(row):
            x = (ci - (span - 1) / 2.0) * 1.28
            pos[ids] = (x, y)

    for ids in singles:
        for sub in subsets:
            if ids[0] in sub and ids in pos and sub in pos:
                x0, y0 = pos[ids]
                x1, y1 = pos[sub]
                ax.add_patch(FancyArrowPatch(
                    (x0, y0 - rh * 0.55), (x1, y1 + rh * 0.62),
                    arrowstyle="-|>", mutation_scale=7, lw=0.55,
                    color="#9aa3ab", alpha=0.55, zorder=1,
                    shrinkA=0, shrinkB=0))

    for ids, (x, y) in pos.items():
        v = verdict[ids]
        face = VERDICT_COLOUR.get(v, "#cccccc")
        edge = "#1a1a1a" if set(ids) == applied and applied else "#6b7278"
        lw = 2.1 if (set(ids) == applied and applied) else 0.7
        ax.add_patch(Ellipse((x, y), rw, rh, facecolor=face, edgecolor=edge,
                             lw=lw, alpha=0.95, zorder=3))
        if len(ids) == 1:
            c = cands[ids[0]]
            mark = TIER_MARK.get(tiers.get(ids[0], ""), "")
            tier = tiers.get(ids[0], "")
            txt = (f"{ids[0]}  {mark} {tier}\n{labels.get(c['operation'], c['operation'][:22])}")
            fs = 6.4
        else:
            txt = "+".join(ids)
            fs = 6.0
        dark = v in ("SURVIVES", "INFEASIBLE", "NOT_REPRESENTABLE")
        ax.text(x, y, txt, ha="center", va="center", fontsize=fs, zorder=4,
                color="white" if dark else "#1a1a1a", linespacing=1.15)
        if cleared[ids] and v != "SURVIVES":
            ax.text(x, y - rh * 0.72, f"clears {len(cleared[ids])}", ha="center",
                    va="top", fontsize=5.2, color=meta_grey, zorder=4)

    ax.set_xlim(min(p[0] for p in pos.values()) - 0.95,
                max(p[0] for p in pos.values()) + 0.95)
    ax.set_ylim(min(p[1] for p in pos.values()) - 0.95, rh + 0.95)
    ax.axis("off")

    handles = [Ellipse((0, 0), 1, 1, facecolor=VERDICT_COLOUR[k], edgecolor="#6b7278",
                       lw=0.6, label=f"{k.replace('_', ' ').lower()} \u2014 {VERDICT_MEANING[k]}")
               for k in ("SURVIVES", "ELIMINATED", "INFEASIBLE", "NOT_REPRESENTABLE",
                         "NOT_ASSESSED") if any(v == k for v in verdict.values())]
    leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.01),
                    frameon=False, fontsize=6, ncol=2, handlelength=1.1,
                    handleheight=0.9, borderaxespad=0.0)
    for t in leg.get_texts():
        t.set_color(meta_grey)

    n_targets = len(record.get("targets", []))
    ax.set_title(
        f"Hypothesis space after epoch {record.get('epoch', 1)}: "
        f"{len(singles)} hypotheses, {len(verdict)} edit sets discharged against "
        f"{n_targets} anomal{'y' if n_targets == 1 else 'ies'}",
        fontsize=9, loc="left", pad=16)
    ax.text(0.0, 1.012,
            f"Tier: {TIER_MARK['T1']} T1 intent-preserving    {TIER_MARK['T2']} T2 "
            f"intent-revealing    {TIER_MARK['T3']} T3 intent-altering.    "
            "An arrow runs from each hypothesis to every edit set containing it; "
            "a bold outline marks the set that was applied.",
            transform=ax.transAxes, fontsize=6.2, color=meta_grey, va="top")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig

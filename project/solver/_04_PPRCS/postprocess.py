"""
postprocess.py
Finished 5/20/26
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio
import h5py


def build_cell_perm(domain, Q, nelx: int, nely: int, Lx: float, Ly: float) -> np.ndarray:
    """
    Build a permutation array that maps row-major grid index to DOLFINx DG-0 DOF index.

    DOLFINx create_rectangle (quad) uses a diagonal cell ordering, not row-major.
    perm[iy * nelx + ix]  =  DOF index for the cell at spatial position (ix, iy).

    Usage:
        rho_spatial = rho_fn.x.array[perm]      # reorder to spatial grid
        rho_grid    = rho_spatial.reshape((nely, nelx))  # then reshape safely
    """
    domain.topology.create_connectivity(2, 0)
    conn = domain.topology.connectivity(2, 0)
    geom = domain.geometry
    dx = Lx / nelx
    dy = Ly / nely
    n_cells = nelx * nely
    perm = np.zeros(n_cells, dtype=int)
    for c in range(n_cells):
        verts = conn.links(c)
        mid = geom.x[verts].mean(axis=0)
        ix = int(round(mid[0] / dx - 0.5))
        iy = int(round(mid[1] / dy - 0.5))
        dof = Q.dofmap.cell_dofs(c)[0]
        perm[iy * nelx + ix] = dof
    return perm


def save_frame(rho_fn, nelx: int, nely: int, iteration: int,
               compliance: float, out_dir: str, perm: np.ndarray) -> str:
    os.makedirs(out_dir, exist_ok=True)
    rho = rho_fn.x.array.copy()
    # DOLFINx uses diagonal cell ordering; perm maps spatial grid index → DOF index
    rho_grid = rho[perm].reshape((nely, nelx))

    fig, ax = plt.subplots(figsize=(nelx / 20, nely / 20), dpi=100)
    ax.imshow(
        1.0 - rho_grid,
        cmap="gray", vmin=0.0, vmax=1.0,
        origin="lower", interpolation="nearest"
    )
    ax.set_title(f"Iter {iteration:03d} | C = {compliance:.4f}", fontsize=8)
    ax.axis("off")
    plt.tight_layout(pad=0.1)

    path = os.path.join(out_dir, f"frame_{iteration:04d}.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def save_gif(frame_paths: list, out_path: str, fps: int = 5):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    frames = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(out_path, frames, fps=fps)

def save_summary_slide(
    rho_history: list,
    compliance_history: list,
    volfrac_history: list,
    change_history: list,
    perm: np.ndarray,
    nelx: int,
    nely: int,
    volfrac_target: float,
    out_dir: str,
    problem_name: str = "",
):
    """
    Save a summary slide PNG with:
      - Initial and final density fields
      - Compliance over iterations (log scale)
      - Volume fraction over iterations
      - Change over iterations
    """
    os.makedirs(out_dir, exist_ok=True)

    iters = np.arange(1, len(compliance_history) + 1)

    rho_initial = rho_history[0][perm].reshape((nely, nelx))
    rho_final   = rho_history[-1][perm].reshape((nely, nelx))

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"Topology Optimization Summary  —  {problem_name}  |  "
        f"{len(compliance_history)} iterations  |  "
        f"Final C = {compliance_history[-1]:.6f}",
        fontsize=13, fontweight="bold", y=0.98
    )

    gs = fig.add_gridspec(
        3, 3,
        hspace=0.45, wspace=0.35,
        left=0.06, right=0.97,
        top=0.92, bottom=0.07
    )

    # ── Row 0-1, Col 0: initial density ──────────────────────────────
    ax_init = fig.add_subplot(gs[0:2, 0])
    ax_init.imshow(1.0 - rho_initial, cmap="gray", vmin=0, vmax=1,
                   origin="lower", interpolation="nearest")
    ax_init.set_title("Initial density  (iter 1)", fontsize=9)
    ax_init.axis("off")

    # ── Row 0-1, Col 1-2: final density ──────────────────────────────
    ax_final = fig.add_subplot(gs[0:2, 1:])
    ax_final.imshow(1.0 - rho_final, cmap="gray", vmin=0, vmax=1,
                    origin="lower", interpolation="nearest")
    ax_final.set_title(f"Final density  (iter {len(rho_history)})", fontsize=9)
    ax_final.axis("off")

    # ── Row 2, Col 0: compliance ──────────────────────────────────────
    ax_c = fig.add_subplot(gs[2, 0])
    ax_c.semilogy(iters, compliance_history, color="#2563eb", linewidth=1.5)
    ax_c.set_xlabel("Iteration", fontsize=8)
    ax_c.set_ylabel("Compliance (log)", fontsize=8)
    ax_c.set_title("Compliance", fontsize=9)
    ax_c.tick_params(labelsize=7)
    ax_c.grid(True, which="both", alpha=0.3)

    # ── Row 2, Col 1: volume fraction ─────────────────────────────────
    ax_v = fig.add_subplot(gs[2, 1])
    ax_v.plot(iters, volfrac_history, color="#16a34a", linewidth=1.5)
    ax_v.axhline(volfrac_target, color="red", linestyle="--",
                 linewidth=1.0, label=f"Target {volfrac_target}")
    ax_v.set_xlabel("Iteration", fontsize=8)
    ax_v.set_ylabel("Volume fraction", fontsize=8)
    ax_v.set_title("Volume Fraction", fontsize=9)
    ax_v.legend(fontsize=7)
    ax_v.tick_params(labelsize=7)
    ax_v.grid(True, alpha=0.3)

    # ── Row 2, Col 2: change ──────────────────────────────────────────
    ax_ch = fig.add_subplot(gs[2, 2])
    ax_ch.plot(iters, change_history, color="#9333ea", linewidth=1.5)
    ax_ch.axhline(0.01, color="red", linestyle="--",
                  linewidth=1.0, label="Tolerance 0.01")
    ax_ch.set_xlabel("Iteration", fontsize=8)
    ax_ch.set_ylabel("Max density change", fontsize=8)
    ax_ch.set_title("Convergence (Change)", fontsize=9)
    ax_ch.legend(fontsize=7)
    ax_ch.tick_params(labelsize=7)
    ax_ch.grid(True, alpha=0.3)

    out_path = os.path.join(out_dir, "summary.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Summary slide saved → {out_path}")
    return out_path
def export_xdmf(nelx: int, nely: int, rho_history: list, perm: np.ndarray, output_dir: str = "_05_OUT"):
    """
    Export density history to XDMF + HDF5 for ParaView animation.

    Parameters
    ----------
    nelx, nely   : mesh element counts
    rho_history  : list of np.ndarray, each shape (nely*nelx,), one per iteration.
                   Each array is rho_fn.x.array.copy() from DG-0 space.
    output_dir   : directory to write topopt.h5 and topopt.xdmf

    DOLFINx cell ordering (create_rectangle, quadrilateral):
        cell_index = j * nelx + i    (j = row from bottom, i = col from left)
    This is row-major / y-fast. The connectivity below must match this.
    """
    os.makedirs(output_dir, exist_ok=True)
    h5_path   = os.path.join(output_dir, "topopt.h5")
    xdmf_path = os.path.join(output_dir, "topopt.xdmf")

    n_nodes_x = nelx + 1
    n_nodes_y = nely + 1
    n_nodes   = n_nodes_x * n_nodes_y
    n_cells   = nelx * nely
    n_iters   = len(rho_history)

    # ------------------------------------------------------------------
    # Node coordinates (physical units match Lx/Ly via element aspect ratio)
    # Node index convention: node(i, j) = j * n_nodes_x + i
    # i in [0, nelx], j in [0, nely]
    # ------------------------------------------------------------------
    coords = np.zeros((n_nodes, 3), dtype=np.float64)
    for j in range(n_nodes_y):
        for i in range(n_nodes_x):
            coords[j * n_nodes_x + i, 0] = i   # x
            coords[j * n_nodes_x + i, 1] = j   # y
            # z = 0 (plane problem)

    # ------------------------------------------------------------------
    # Quad connectivity — must match DOLFINx DG-0 cell ordering:
    #     cell_index = j * nelx + i
    #
    # For element at grid position (i, j):
    #   bottom-left  node: j       * n_nodes_x + i
    #   bottom-right node: j       * n_nodes_x + (i+1)
    #   top-right    node: (j+1)   * n_nodes_x + (i+1)
    #   top-left     node: (j+1)   * n_nodes_x + i
    #
    # XDMF quad winding: counter-clockwise from bottom-left
    # ------------------------------------------------------------------
    connectivity = np.zeros((n_cells, 4), dtype=np.int32)
    for j in range(nely):          # row index (y direction)
        for i in range(nelx):      # col index (x direction)
            eid = j * nelx + i     # matches DOLFINx create_rectangle ordering
            connectivity[eid] = [
                j       * n_nodes_x + i,        # bottom-left
                j       * n_nodes_x + (i + 1),  # bottom-right
                (j + 1) * n_nodes_x + (i + 1),  # top-right
                (j + 1) * n_nodes_x + i,         # top-left
            ]

    # ------------------------------------------------------------------
    # Write HDF5
    # ------------------------------------------------------------------
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("nodes",        data=coords)
        h5.create_dataset("connectivity", data=connectivity)
        for it, rho in enumerate(rho_history):
            # Reorder from DOLFINx diagonal DOF order to spatial row-major order
            # so density[eid] matches the cell at grid position eid = j*nelx+i
            h5.create_dataset(
                f"density/iter_{it:04d}",
                data=rho[perm].astype(np.float64)
            )

    # ------------------------------------------------------------------
    # Write XDMF
    # ------------------------------------------------------------------
    with open(xdmf_path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd">\n')
        f.write('<Xdmf Version="2.0">\n<Domain>\n')
        f.write('  <Grid Name="TopOpt" GridType="Collection" CollectionType="Temporal">\n')

        for it in range(n_iters):
            f.write(f'    <Grid Name="iter_{it:04d}" GridType="Uniform">\n')
            f.write(f'      <Time Value="{it}"/>\n')
            f.write(f'      <Topology TopologyType="Quadrilateral" NumberOfElements="{n_cells}">\n')
            f.write(f'        <DataItem Format="HDF" DataType="Int" Dimensions="{n_cells} 4">\n')
            f.write(f'          topopt.h5:/connectivity\n')
            f.write(f'        </DataItem>\n')
            f.write(f'      </Topology>\n')
            f.write(f'      <Geometry GeometryType="XYZ">\n')
            f.write(f'        <DataItem Format="HDF" DataType="Float" Dimensions="{n_nodes} 3">\n')
            f.write(f'          topopt.h5:/nodes\n')
            f.write(f'        </DataItem>\n')
            f.write(f'      </Geometry>\n')
            f.write(f'      <Attribute Name="density" AttributeType="Scalar" Center="Cell">\n')
            f.write(f'        <DataItem Format="HDF" DataType="Float" Dimensions="{n_cells}">\n')
            f.write(f'          topopt.h5:/density/iter_{it:04d}\n')
            f.write(f'        </DataItem>\n')
            f.write(f'      </Attribute>\n')
            f.write(f'    </Grid>\n')

        f.write('  </Grid>\n</Domain>\n</Xdmf>\n')

    print(f"Exported {n_iters} iterations → {xdmf_path}")


def print_iteration_report(iteration: int, compliance: float,
                           volfrac_actual: float, change: float):
    print(
        f"Iter {iteration:4d} | "
        f"C = {compliance:12.6f} | "
        f"Vol = {volfrac_actual:.4f} | "
        f"Change = {change:.6f}"
    )
def save_final_density(rho_fn, nelx, nely, out_dir, perm):
    os.makedirs(out_dir, exist_ok=True)
    rho = rho_fn.x.array.copy()
    rho_grid = rho[perm].reshape((nely, nelx))

    fig, ax = plt.subplots(figsize=(nelx / 20, nely / 20), dpi=150)
    ax.imshow(
        1.0 - rho_grid,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        origin="lower",
        interpolation="nearest"
    )
    ax.set_title("Final density")
    ax.axis("off")
    plt.tight_layout(pad=0.1)

    path = os.path.join(out_dir, "final_density.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
# Vendored ParaLeSTO build sources

`pyparalesto/` contains the Python extension build sources from the project's
MFRTO checkout, copied on 2026-10-02. Upstream is
[M2DO ParaLeSTO](https://gitlab.com/m2dO1/paralesto), version 1.0.0 in `setup.py`.
The Apache-2.0 license and upstream attribution are retained. The source inventory
and SHA-256 hashes are in `pyparalesto-source.json`.

Only sources required by the three Python extensions, package modules, build
metadata, README and license are included. Upstream examples, compiled binaries,
build directories and its legacy FEniCS environment are excluded. The Dockerfile
builds a non-editable installation against Eigen, NLopt and GLPK in the image.
FEM uses the separately pinned DOLFINx 0.7.0 / PETSc 3.19.6 stack; ParaLeSTO's
Python extensions do not require legacy FEniCS.

cat >> README.md <<'EOF'

## Repository structure

- `project/apps/`: interactive and batch entry points
- `project/parser/`: natural-language parsing, schema, and provenance
- `project/topopt/`: deterministic DOLFINx FEM and SIMP/MMA implementation
- `project/verification/`: hard validation gate and numerical V&V
- `project/llm/`: evidence-limited critic and deferred steering logic
- `project/experiments/`: prompt corpus and manual parser experiments
- `project/tools/`: repository and artifact-management utilities
- `artifacts/`: generated runs, verification records, and V&V evidence

## Primary commands

Run numerical verification:

```bash
/dolfinx-env/bin/python -m project.verification.run_suite
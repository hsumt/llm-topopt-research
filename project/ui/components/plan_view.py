"""The exact plan an approval records: solver blocks, inputs, budget and hash."""
import streamlit as st

from project.ui.components import spec_view
from project.ui.components.spec_view import capital, number, readable, words

_SUFFIXES = (("_m", "m"), ("_pa", "Pa"), ("_n", "N"))


def _parameter(name: str, value) -> str:
    for suffix, unit in _SUFFIXES:
        if name.endswith(suffix):
            text, shown = readable(value, unit)
            return f"{name[:-len(suffix)]} = {text} {shown}"
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return f"{name} = {', '.join(value)}"
    return f"{name} = {number(value)}"


def block_table(block_config: list[dict]) -> None:
    """The solver blocks; interim: mapped from the 2D L-bracket solver inputs (api._block_config)."""
    st.table([{"Kind": capital(words(block.get("kind"))) or "",
               "Block": block.get("name", ""),
               "Parameters": "; ".join(_parameter(k, v) for k, v in (block.get("params") or {}).items())}
              for block in block_config], hide_index=True)
    st.caption("Interim: these blocks are mapped from the current L-bracket solver inputs.")


def budget_text(budget: dict) -> str:
    iterations, ranks = budget.get("max_iterations"), budget.get("mpi_ranks")
    return (f"at most {iterations} optimization iteration{'s' if iterations != 1 else ''} "
            f"on {ranks} MPI process{'es' if ranks != 1 else ''}")


def render(entry) -> None:
    st.markdown("**Solver blocks**")
    block_table(entry.block_config)
    st.markdown(f"**Budget:** {budget_text(entry.budget)}")
    with st.expander("Exact solver inputs"):
        st.json(entry.solver_inputs, expanded=True)
    with st.expander("Specification"):
        spec_view.render(entry.spec)
    st.caption(f"Plan hash (SHA-256): `{entry.plan_hash}`")

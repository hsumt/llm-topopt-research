"""A copy of an approved formulation: where it came from, and what it changes against the approved plan."""
import streamlit as st

from project.ui import routes
from project.ui.components import job_status
from project.ui.components.spec_view import md


def origin(parent: dict) -> None:
    """One line naming the approved formulation this one was copied from, with links to it and its approval."""
    approved = (f"approved by {md(parent['approver'])} on {job_status.when(parent['approved_at'])}"
                if parent.get("approver") else "approved")
    run = f" · run {parent['run']}" if parent.get("run") else ""
    st.info(f"Copy of **{md(parent.get('title') or 'an approved formulation')}**, {approved}{run}. "
            "That formulation, its approval and its runs stay as they are.", icon=":material/edit_document:")
    with st.container(horizontal=True):
        st.page_link(routes.FORMULATION, label="Open the original", icon=":material/open_in_new:",
                     query_params={"fid": parent["fid"]})
        if parent.get("approval_id"):
            st.page_link(routes.APPROVAL, label="Its approval record", icon=":material/receipt_long:",
                         query_params={"aid": parent["approval_id"]})


def _table(rows: list[dict]) -> None:
    st.dataframe([{"Field": r["field"], "Approved": r["approved"], "Now": r["now"]} for r in rows],
                 hide_index=True, width="stretch")


def render(changes: dict | None, *, expanded: bool = True, editable: bool = False) -> None:
    """The copy's changes to the specification and to the solver inputs, against the approved plan.
    editable: the page has the change panel below, so the text can point to it."""
    if changes is None:
        return
    spec, inputs = changes["spec"], changes["solver_inputs"]
    # Solver inputs count too: they differ with the same spec when the code changed after the approval.
    parts = [] if spec is None else [f"{len(rows)} in the {name}" for name, rows in
                                     (("specification", spec), ("solver inputs", inputs or [])) if rows]
    title = ("Changes from the approved plan" if spec is None else
             f"Changes from the approved plan: {', '.join(parts)}" if parts else "No changes from the approved plan yet")
    with st.expander(title, expanded=expanded and bool(parts), icon=":material/difference:"):
        if spec is None:
            st.caption("The approval record of the original cannot be read, so the changes cannot be listed.")
            return
        if not spec:
            st.caption("The specification is still the approved one."
                       + (" Edit values or describe a change below." if editable else ""))
        else:
            st.markdown("**Specification**")
            _table(spec)
        if inputs is None:
            st.caption("Solver inputs: not built while the pre-solve check finds blocking issues.")
        elif inputs:
            st.markdown("**Solver inputs** (SI units, as the solver receives them)")
            _table(inputs)
        elif spec:
            st.caption("The solver inputs are the approved ones: the changes do not reach the solver.")
        if inputs and not spec:
            st.caption("The specification is the approved one, but the solver inputs the current code builds "
                       "from it differ from the approved plan: the code changed after that approval.")

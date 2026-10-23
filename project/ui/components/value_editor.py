"""The value editor: set a formulation's stated values directly, without a model call.

Rows come from api.editable_values; api.edit_values applies the changed ones and the spec is then
checked and reviewed again. Values are edited as text: a number, a list in brackets, or words.
"""
import json

import pandas as pd
import streamlit as st

ROW_PX = 35   # the data editor's default row height


def text(row: dict) -> str:
    """A row's value as the editor shows it."""
    return row["value"] if isinstance(row["value"], str) else json.dumps(row["value"])


def render(fid: str, rows: list[dict], round_index: int) -> dict[str, tuple[str, str]] | None:
    """The table of stated values. Returns {field: (value text, unit text)} for the rows that differ from
    the spec once the engineer submits (empty if none differ), or None before that."""
    if not rows:
        st.caption("The specification states no values that can be set here yet. Answer the questions, "
                   "or describe the change in words.")
        return None
    table = pd.DataFrame({"Field": [row["label"] for row in rows], "Value": [text(row) for row in rows],
                          "Unit": [row["unit"] or "" for row in rows]})
    # One form per round: an applied edit starts a new round, so the table restarts from the new spec.
    with st.form(f"values_{fid}_{round_index}", border=False):
        edited = st.data_editor(
            table, key=f"value_table_{fid}_{round_index}", hide_index=True, num_rows="fixed", disabled=["Field"],
            width="stretch", height=ROW_PX * (len(rows) + 1) + 3,   # every row in view, no inner scroll
            column_config={
                "Field": st.column_config.TextColumn("Field", width="medium"),
                "Value": st.column_config.TextColumn(
                    "Value", width="large", required=True,
                    help="A number, a list of numbers in brackets like the one shown, or words for a direction."),
                "Unit": st.column_config.TextColumn("Unit", width="small",
                                                    help="Leave empty for a value without a unit."),
            })
        submitted = st.form_submit_button("Apply and check", type="primary", icon=":material/fact_check:",
                                          help="Applies the changed values, then checks and reviews the "
                                               "specification again. New questions appear on this page.")
    if not submitted:
        return None
    changes = {}
    for row, value, unit in zip(rows, edited["Value"], edited["Unit"]):
        value = "" if value is None or (isinstance(value, float) and value != value) else str(value)
        unit = "" if unit is None or (isinstance(unit, float) and unit != unit) else str(unit)
        if value.strip() != text(row).strip() or unit.strip() != (row["unit"] or "").strip():
            changes[row["field"]] = (value, unit)
    return changes

"""Model calls and tokens spent on a formulation."""
import streamlit as st

STEPS = {"parse": "Parsing", "review": "Reviewer check", "resolve": "Applying answers", "clarify": "Explaining a question"}


def render(usage: list) -> None:
    if not usage:
        st.caption("No model calls were made for this formulation.")
        return
    cached = sum(1 for u in usage if u.cache_hit)
    columns = st.columns(4)
    columns[0].metric("Input tokens", f"{sum(u.input_tokens for u in usage):,}")
    columns[1].metric("Output tokens", f"{sum(u.output_tokens for u in usage):,}")
    columns[2].metric("Model calls", len(usage))
    columns[3].metric("From local cache", f"{cached} of {len(usage)}")
    st.dataframe(
        [{"Step": STEPS.get(u.step, u.step), "Model": u.model, "Input tokens": u.input_tokens,
          "Output tokens": u.output_tokens, "From cache": u.cache_hit} for u in usage],
        hide_index=True,
        column_config={"From cache": st.column_config.CheckboxColumn(
            "From cache", help="Reused a saved response; no request was sent.")},
    )
    if cached:
        st.caption("Cached calls reused a saved response, so no request was sent and no tokens were spent.")

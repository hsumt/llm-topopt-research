"""New problem: describe a design problem and start a formulation."""
import streamlit as st

from project.ui import api, routes


def _fill_example() -> None:
    example = api.EXAMPLES.get(st.session_state.get("new_example"))
    if example:
        st.session_state.new_brief = example["request"]
        st.session_state.new_context = example["context"] or ""
        st.session_state.new_filled = st.session_state.get("new_example")


def _insert_outline() -> None:
    st.session_state.new_brief = api.with_outline(st.session_state.get("new_brief") or "")


if st.session_state.pop("new_started", False):
    st.session_state.new_brief = st.session_state.new_context = ""
    st.session_state.pop("new_filled", None)
# A benchmark opened from the Benchmarks page: its documented inputs as a brief, ready to change.
opened = api.benchmark_example(st.query_params.get("benchmark"))
if opened:
    st.session_state.new_example = opened
    _fill_example()
if "benchmark" in st.query_params:
    del st.query_params["benchmark"]

st.title("New problem")
st.write("Describe the problem in your own words. Leave out what you do not know; "
         "the system asks instead of guessing.")

with st.container(horizontal=True, vertical_alignment="bottom"):
    st.selectbox("Start from a suggested problem (optional)", list(api.EXAMPLES), index=None,
                 placeholder="Choose a suggested problem", key="new_example", width=480)
    st.button("Fill in", on_click=_fill_example, disabled=not st.session_state.get("new_example"),
              help="Replaces the design problem and references below with the chosen problem.")
if st.session_state.get("new_filled") == api.BENCHMARK_EXAMPLE:
    st.caption("Filled in with the benchmark's documented inputs. Change anything, then start it as a new problem: "
               "the system reads, checks and reviews it like any brief. To run the benchmark exactly as documented, "
               "with no model calls, use the Benchmarks page.")

with st.container(horizontal=True, vertical_alignment="bottom"):
    st.markdown("**Design problem**")
    st.button("Insert outline", on_click=_insert_outline, icon=":material/format_list_bulleted:",
              help="Adds section headings to the box. Fill in what you know and leave the rest empty; "
                   "empty headings are ignored.")
headings = " · ".join(label for label, _ in api.OUTLINE)
brief = st.text_area("Design problem", key="new_brief", height=260, persist_state="session",
                     label_visibility="collapsed",
                     placeholder=f"Write in any order. Useful topics: {headings}.")

with st.expander("References and conventions (optional)", expanded=bool(st.session_state.get("new_context"))):
    context = st.text_area("References and conventions", key="new_context", height=110, persist_state="session",
                           label_visibility="collapsed",
                           placeholder="Coordinate axes and origin, the drawing or CAD that sets the dimensions, "
                                       "material datasheets, interfaces with other parts.")

st.caption("Method: level set (ParaLeSTO, provisional), the only method connected. "
           "Its settings appear on the approval page.")

if st.button("Start formulation", type="primary", icon=":material/play_arrow:"):
    try:
        fid = api.start_formulation(brief or "", context)
    except (ValueError, RuntimeError) as error:
        st.error(str(error))
    else:
        st.session_state.new_started = True
        st.switch_page(routes.FORMULATION, query_params={"fid": fid})

st.caption(f"Uses {api.MODEL or '(model not set)'} on {api.PROVIDER_NAME}. Each formulation runs in the background, so you can start several and track them in Jobs.")

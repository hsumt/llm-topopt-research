"""Process flow: how a formulation went from brief to approved plan, step by step."""
import streamlit as st

from project.ui import api, routes
from project.ui.components import flow_canvas, job_status

st.title("Process flow")
st.write("Every pre-solve step of a formulation, from the brief to the approved plan. "
         "Click a symbol to see exactly what ran; press Play to step through it.")

formulations = [f for f in api.list_formulations()]
fid = st.query_params.get("fid")
with st.container(horizontal=True, vertical_alignment="bottom"):
    options = {f.fid: f for f in formulations}
    choice = st.selectbox("Formulation", list(options), index=list(options).index(fid) if fid in options else None,
                          format_func=lambda k: f"{options[k].title} · {job_status.label(options[k].state)}"
                                                 f"{' · simulated' if options[k].origin == 'simulation' else ''}",
                          placeholder="Choose a formulation", width=560)
    if st.button("Create a simulated run", icon=":material/slideshow:",
                 help="Runs the real pre-solve code on scripted model replies. No API call is made."):
        st.query_params["fid"] = api.start_simulation()
        st.rerun()
if choice and choice != fid:
    st.query_params["fid"] = choice
    st.rerun()

if not fid or fid not in options:
    st.info("Choose a formulation, or create a simulated run to present the flow without calling a model.")
    st.stop()


@st.fragment(run_every=3)
def _watch(fid: str, version: tuple) -> None:
    if api.flow_version(fid) != version:
        st.rerun()


live = options[fid].state == "working" or api.run_actions(fid)["active"]
if live:
    # Redraw only when a step or the run changes state, so the canvas stays usable meanwhile.
    _watch(fid, api.flow_version(fid))
    st.caption(":material/autorenew: Updates as steps and the run progress.")
flow = api.flow(fid)
if not flow["nodes"] or len(flow["nodes"]) <= 1:
    st.info("This formulation has no recorded trace yet. Traces start with steps run after tracing was added.")
    st.stop()
flow_canvas.render(flow, fid)
st.caption("Drag to pan, scroll to zoom. Purple outlines are model steps, black are code, amber are the engineer. "
           "Model reasoning shown here is the model's own output, not verified fact.")
st.page_link(routes.FORMULATION, label="Open this formulation", icon=":material/description:", query_params={"fid": fid})

"""Process flow: how a formulation went from brief to run results, step by step, live while it runs."""
import streamlit as st

from project.ui import api, flow_model, routes
from project.ui.components import flow_canvas, job_status

st.title("Process flow")
st.write("Every step of a formulation, from the brief through approval to its run and results. "
         "Click a symbol to see exactly what ran; press Play to step through it.")

formulations = api.list_formulations()
fid = st.query_params.get("fid")
options = {f.fid: f for f in formulations}
choice = st.selectbox("Formulation", list(options), index=list(options).index(fid) if fid in options else None,
                      format_func=lambda k: f"{options[k].title} · {job_status.label(options[k].state)}",
                      placeholder="Choose a formulation", width=560)
if choice and choice != fid:
    st.query_params["fid"] = choice
    st.rerun()

if not fid or fid not in options:
    st.info("Choose a formulation.")
    st.stop()

flow = flow_model.flow(fid)
if len(flow["nodes"]) <= 1 and not flow["live"]:
    st.info("This formulation has no recorded steps yet.")
    st.stop()
flow_canvas.show(fid, fid, flow=flow)
st.caption("Drag to pan, scroll to zoom. Purple outlines are model steps, neutral ones are code, amber ones are the engineer; "
           "the pulsing step is running now and dashed steps are still to come. "
           "Model reasoning shown here is the model's own output, not verified fact.")
st.page_link(routes.FORMULATION, label="Open this formulation", icon=":material/description:", query_params={"fid": fid})

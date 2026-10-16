"""Batch: choose several ready formulations and review them for one approval."""
import hashlib

import streamlit as st

from project.ui import api, routes
from project.ui.components import job_status

st.title("Batch")
st.write("Select the ready formulations to approve together, then review them.")

ready = [f for f in api.list_formulations() if f.state == "ready"]

if not ready:
    st.info("No formulation is ready for approval. A formulation becomes ready once every "
            "decision and value it needs is settled.")
    with st.container(horizontal=True):
        st.page_link(routes.BENCHMARKS, label="Use a benchmark", icon=":material/science:")
        st.page_link(routes.NEW_PROBLEM, label="Describe a new problem", icon=":material/edit_note:")
        st.page_link(routes.JOBS, label="Formulations waiting for your input", icon=":material/list_alt:",
                     query_params={"show": "needs_input"})
else:
    rows = [{"Title": f.title, "Source": job_status.origin(f.origin)[0], "Updated": job_status.when(f.updated_at),
             "Id": f.fid[:8]} for f in ready]
    # The key follows the list, so a changed list starts with a clean selection.
    listing = hashlib.sha1(",".join(f.fid for f in ready).encode()).hexdigest()[:12]
    event = st.dataframe(rows, hide_index=True, on_select="rerun", selection_mode="multi-row",
                         key=f"batch_{listing}", column_config={"Id": st.column_config.TextColumn(width="small")})
    selected = [ready[i].fid for i in event.selection.rows]
    label = f"Review {len(selected)} selected for approval" if selected else "Review selected for approval"
    if st.button(label, type="primary", disabled=not selected, icon=":material/fact_check:"):
        st.switch_page(routes.APPROVAL, query_params={"fids": ",".join(selected)})

benchmarks = api.list_benchmarks()
if benchmarks:
    with st.expander("Add a benchmark to this list", icon=":material/add:"):
        with st.container(horizontal=True, vertical_alignment="bottom"):
            choice = st.selectbox("Benchmark", benchmarks, format_func=lambda b: b.title, key="batch_benchmark")
            add = st.button("Add", key="batch_add")
        if add:
            try:
                api.start_benchmark(choice.id)
            except (ValueError, RuntimeError) as error:
                st.error(str(error))
            else:
                st.rerun()

st.caption("Sweeps (one plan expanded into several variants) are not built yet.")

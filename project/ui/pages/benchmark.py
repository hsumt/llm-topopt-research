"""Benchmarks: start a ready formulation from a documented case."""
import streamlit as st

from project.ui import api, routes

st.title("Benchmarks")
st.write("Pick a documented case to go straight to approval, with no model calls, or open it as a new problem "
         "to change it first. Each use creates a new formulation.")

benchmarks = api.list_benchmarks()
if not benchmarks:
    st.info("No benchmarks are defined yet.")

for benchmark in benchmarks:
    with st.container(border=True):
        st.subheader(benchmark.title)
        st.write(benchmark.description)
        st.caption(f"Benchmark id: {benchmark.id}")
        with st.container(horizontal=True):
            use = st.button("Use this benchmark", key=f"benchmark_{benchmark.id}", type="primary",
                            icon=":material/arrow_forward:",
                            help="Exactly the documented inputs, straight to approval. No model calls.")
            if benchmark.example:
                st.page_link(routes.NEW_PROBLEM, label="Edit as a new problem", icon=":material/edit_note:",
                             query_params={"benchmark": benchmark.id},
                             help="Opens New problem filled in with these inputs as a brief. Change anything, then "
                                  "start it: the system reads, checks and reviews it like any new problem.")
        if use:
            try:
                fid = api.start_benchmark(benchmark.id)
            except (ValueError, RuntimeError) as error:
                st.error(str(error))
            else:
                st.switch_page(routes.APPROVAL, query_params={"fids": fid})

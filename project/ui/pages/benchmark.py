"""Benchmarks: start a ready formulation from a documented case."""
import streamlit as st

from project.ui import api, routes

st.title("Benchmarks")
st.write("Pick a documented case to go straight to approval. "
         "Each use creates a new formulation; no model calls are made.")

benchmarks = api.list_benchmarks()
if not benchmarks:
    st.info("No benchmarks are defined yet.")

for benchmark in benchmarks:
    with st.container(border=True):
        st.subheader(benchmark.title)
        st.write(benchmark.description)
        st.caption(f"Benchmark id: {benchmark.id}")
        if st.button("Use this benchmark", key=f"benchmark_{benchmark.id}", type="primary",
                     icon=":material/arrow_forward:"):
            try:
                fid = api.start_benchmark(benchmark.id)
            except (ValueError, RuntimeError) as error:
                st.error(str(error))
            else:
                st.switch_page(routes.APPROVAL, query_params={"fids": fid})

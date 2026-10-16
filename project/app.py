"""ATO Streamlit entry point. Pages live in project/ui/pages and call project.ui.api (and project.ui.flow_model)."""
import streamlit as st

from project.ui import routes

st.set_page_config(page_title="ATO", page_icon="⚙️", layout="wide")

st.navigation([
    st.Page(routes.HOME, title="Home", icon=":material/home:", default=True),
    st.Page(routes.NEW_PROBLEM, title="New problem", icon=":material/edit_note:"),
    st.Page(routes.BENCHMARKS, title="Benchmarks", icon=":material/science:"),
    st.Page(routes.BATCH, title="Batch", icon=":material/stacks:"),
    st.Page(routes.JOBS, title="Jobs", icon=":material/list_alt:"),
    st.Page(routes.FLOW, title="Process flow", icon=":material/account_tree:"),
    st.Page(routes.FORMULATION, title="Formulation", visibility="hidden"),
    st.Page(routes.APPROVAL, title="Approval", visibility="hidden"),
]).run()

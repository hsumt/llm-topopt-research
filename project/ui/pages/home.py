"""Home: what ATO does and where to start."""
from collections import Counter

import streamlit as st

from project.ui import api, routes
from project.ui.components import job_status

CARDS = [
    ("Run a benchmark", ":material/science:", routes.BENCHMARKS, "Choose a benchmark",
     "Use a documented case: no model calls, straight to approval."),
    ("New problem", ":material/edit_note:", routes.NEW_PROBLEM, "Describe a problem",
     "Describe a design problem; the system builds a specification and asks for what is missing. "
     "Runs in the background."),
    ("Batch", ":material/stacks:", routes.BATCH, "Open batch",
     "Approve several ready formulations together."),
    ("Jobs", ":material/list_alt:", routes.JOBS, "Open jobs",
     "Formulations in progress, waiting for you, ready, or approved."),
]

st.title("Agentic Topology Optimization")
st.write("Turn a design problem into an explicit specification and approved solver inputs. "
         "Choose where to start below.")

for row in (CARDS[:2], CARDS[2:]):
    for column, (title, icon, page, action, description) in zip(st.columns(2), row):
        with column, st.container(border=True, height="stretch"):
            st.subheader(title, icon=icon)
            st.write(description)
            kind = "primary" if page == routes.NEW_PROBLEM else "secondary"
            if st.button(action, key=f"home_{page}", type=kind, icon=":material/arrow_forward:"):
                st.switch_page(page)

st.subheader("Your formulations")
counts = Counter(f.state for f in api.list_formulations())
if not counts:
    st.caption("None yet. Run a benchmark or describe a new problem to create one.")
else:
    with st.container(horizontal=True, gap="medium"):
        for state in api.STATES:
            if counts[state]:
                with st.container(width="content", gap="xsmall"):
                    job_status.badge(state)
                    st.page_link(routes.JOBS, label=f"{counts[state]} in Jobs",
                                 icon=":material/arrow_forward:", query_params={"show": state})

st.caption("Approved plans run from their approval record or formulation page. "
           "Post-solve diagnosis is not connected yet.")

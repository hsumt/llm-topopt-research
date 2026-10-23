"""Why a formulation step failed and what to do about it."""
import streamlit as st

from project.ui.components.spec_view import md
from project.ui.components.usage_view import STEPS
REQUEST_STATES = {
    "not_sent": "No request reached the model provider.",
    "attempted": "A request to the model provider was attempted, but no response came back.",
    "response_received": "The model provider received the request and responded.",
    "unknown": "It is not known whether a request reached the model provider.",
}
KEY_CODES = ("missing_api_key", "authentication_failed", "missing_model", "unknown_provider")


def _key_help() -> None:
    with st.expander("How the .env settings reach the app"):
        st.markdown(
            "The API key and model settings are read from the repository's `.env` file by "
            "`./docker/compose` when the container is created. Editing `.env` does not reach a "
            "running container: recreate it from the repository root, then press Retry.")
        st.code("./docker/compose up -d --force-recreate", language="bash")
        st.warning("Recreating the container stops any formulation that is still working. Those "
                   "show as failed afterwards and can be retried.", icon=":material/warning:")


def render(failure, error: str | None) -> None:
    if error:
        st.error(error, icon=":material/error:")
    if failure is not None:
        stage = STEPS.get(failure.stage, failure.stage)
        st.error(md(failure.message), title=f"{stage} failed", icon=":material/error:")
        st.caption(f"{REQUEST_STATES.get(failure.request_state, REQUEST_STATES['unknown'])} "
                   f"Error code: `{failure.code}`.")
        st.markdown(f"**What to do:** {md(failure.next_step)}")
        if failure.code in KEY_CODES:
            _key_help()
    if not error and failure is None:
        st.error("This step failed without a recorded reason. Retry the step.", icon=":material/error:")

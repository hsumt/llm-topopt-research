"""Safe, actionable failures at the model boundary.

Provider exceptions can contain request bodies and credentials. Never render or
persist them; classify only their status code, SDK exception type and error code instead.
This module deliberately does not import either provider SDK.
"""
from typing import Literal


RequestState = Literal["not_sent", "attempted", "response_received", "unknown"]


class ModelCallError(Exception):
    """A controlled diagnostic suitable for the UI and saved run history."""

    def __init__(
        self,
        code: str,
        message: str,
        next_step: str,
        request_state: RequestState = "unknown",
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.next_step = next_step
        self.request_state = request_state
        self.retryable = retryable


def classify_error(error: Exception) -> ModelCallError:
    """Preserve safe diagnostics; never stringify an unrecognized exception."""
    if isinstance(error, ModelCallError):
        return error
    return ModelCallError(
        "unknown_error",
        "The AI step failed before a usable result could be saved.",
        "Retry the AI step. If it continues to fail, check the application setup; "
        "adding engineering details will not resolve this error.",
        retryable=True,
    )


def classify_provider_error(
    error: Exception, *, provider: str = "Anthropic", key_var: str = "ANTHROPIC_API_KEY",
    model_var: str = "ANTHROPIC_MODEL",
) -> ModelCallError:
    """Classify a model call failure without reading its body or message.

    provider names the service in the message; key_var and model_var are the variables it reads.
    """
    if isinstance(error, ModelCallError):
        return error
    status = getattr(error, "status_code", None)
    types = {cls.__name__ for cls in type(error).__mro__}
    if status == 401 or "AuthenticationError" in types:
        return ModelCallError(
            "authentication_failed", f"{provider} rejected the API credentials.",
            f"Supply a valid {key_var} in the container runtime environment, "
            "recreate the app container, then retry. Do not enter a key in the design brief.",
            "response_received",
        )
    if status == 403 or "PermissionDeniedError" in types:
        return ModelCallError(
            "permission_denied", f"{provider} denied access to the requested model or workspace.",
            f"Check the account's access to the model and {model_var}, then retry.",
            "response_received",
        )
    if status == 429 or "RateLimitError" in types:
        return ModelCallError(
            "rate_limited", f"{provider} returned a rate or usage limit response.",
            "Wait and retry. If it persists, check the API account's rate and usage limits.",
            "response_received", True,
        )
    if "APITimeoutError" in types or isinstance(error, TimeoutError):
        return ModelCallError(
            "request_timeout", f"The {provider} request timed out before a response was available.",
            "Retry the AI step. If it repeats, check the container's network connection.",
            "attempted", True,
        )
    if "APIConnectionError" in types or isinstance(error, ConnectionError):
        return ModelCallError(
            "connection_failed", f"The app could not complete its connection to {provider}.",
            "Check the container's network, proxy and TLS configuration, then retry.",
            "attempted", True,
        )
    if isinstance(status, int) and status >= 500:
        return ModelCallError(
            "provider_unavailable", f"{provider} returned a server error.",
            "Wait briefly and retry the AI step.", "response_received", True,
        )
    if status == 404 or "NotFoundError" in types:
        return ModelCallError(
            "model_unavailable", f"{provider} could not find the requested model or API resource.",
            f"Check {model_var} and the API account's access, recreate the app container "
            "if its configuration changes, then retry.", "response_received",
        )
    # Azure OpenAI rejects a prompt its content filter blocks with a 400 whose error code says so.
    if getattr(error, "code", None) == "content_filter":
        return ModelCallError(
            "content_filtered", f"{provider}'s content filter blocked the request.",
            "Check the brief for wording a content filter could misread, then retry; "
            "this is not a missing engineering decision.",
            "response_received",
        )
    if status in (400, 413, 422) or "BadRequestError" in types:
        return ModelCallError(
            "invalid_request", f"{provider} rejected the app's model request.",
            "Check the configured model and API compatibility. If the brief is unusually "
            "large, shorten it before retrying; this is not a missing engineering decision.",
            "response_received",
        )
    failure = classify_error(error)
    failure.request_state = "response_received" if isinstance(status, int) else "attempted"
    return failure

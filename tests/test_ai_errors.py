"""Provider boundary regressions. All credentials and network clients are mocked."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from project import ai
from project.ai_errors import ModelCallError, classify_error, classify_provider_error
from project.models import ProblemSpec, Review


PRIVATE = "PRIVATE_PROVIDER_BODY_MUST_NOT_BE_RENDERED"


def response(text='{"name":"Parsed bracket","problem_kind":"optimization"}', **changes):
    values = {
        "content": [SimpleNamespace(type="text", text=text)],
        "stop_reason": "end_turn",
        "usage": SimpleNamespace(input_tokens=12, output_tokens=20),
    }
    values.update(changes)
    return SimpleNamespace(**values)


def call_model():
    return ai._call_model(step="parse", prompt="test instructions", payload={"request": "brief"},
                          output_type=ProblemSpec, max_tokens=100)


class ErrorClassificationTests(unittest.TestCase):
    def test_provider_status_codes_are_actionable_and_sanitized(self):
        cases = {
            401: ("authentication_failed", False),
            403: ("permission_denied", False),
            404: ("model_unavailable", False),
            429: ("rate_limited", True),
            400: ("invalid_request", False),
            500: ("provider_unavailable", True),
            529: ("provider_unavailable", True),
        }
        for status, (code, retryable) in cases.items():
            with self.subTest(status=status):
                provider_error = RuntimeError(PRIVATE)
                provider_error.status_code = status
                failure = classify_provider_error(provider_error)
                self.assertEqual(failure.code, code)
                self.assertEqual(failure.retryable, retryable)
                self.assertEqual(failure.request_state, "response_received")
                self.assertTrue(failure.next_step)
                self.assertNotIn(PRIVATE, str(failure))
                self.assertNotIn(PRIVATE, repr(vars(failure)))

    def test_connection_and_timeout_cannot_claim_provider_receipt(self):
        for name, code in (("APITimeoutError", "request_timeout"),
                           ("APIConnectionError", "connection_failed")):
            with self.subTest(name=name):
                failure = classify_provider_error(type(name, (Exception,), {})(PRIVATE))
                self.assertEqual(failure.code, code)
                self.assertEqual(failure.request_state, "attempted")
                self.assertTrue(failure.retryable)

    def test_unknown_error_is_never_stringified(self):
        class UnsafeError(Exception):
            def __str__(self):
                raise AssertionError("Exception text must never be read")
        failure = classify_error(UnsafeError())
        self.assertEqual(failure.code, "unknown_error")
        self.assertEqual(failure.request_state, "unknown")
        self.assertEqual(classify_provider_error(UnsafeError()).request_state, "attempted")
        self.assertIs(classify_error(failure), failure)


class ClientConfigurationTests(unittest.TestCase):
    def test_missing_configuration_never_constructs_or_calls_sdk(self):
        sdk = types.ModuleType("anthropic")
        sdk.Anthropic = Mock()
        with patch.object(ai.os, "getenv", return_value=None), \
             patch.dict(sys.modules, {"anthropic": sdk}), \
             self.assertRaises(ModelCallError) as raised:
            ai._client()
        self.assertEqual(raised.exception.code, "missing_api_key")
        self.assertEqual(raised.exception.request_state, "not_sent")
        sdk.Anthropic.assert_not_called()

    def test_sdk_is_optional_until_a_configured_call_and_wait_is_bounded(self):
        sdk = types.ModuleType("anthropic")
        sdk.Anthropic = Mock()
        with patch.object(ai.os, "getenv", return_value="synthetic-test-value"), \
             patch.dict(sys.modules, {"anthropic": sdk}):
            self.assertIs(ai._client(), sdk.Anthropic.return_value)
        self.assertEqual(sdk.Anthropic.call_args.kwargs["timeout"], 45.0)
        self.assertEqual(sdk.Anthropic.call_args.kwargs["max_retries"], 0)

    def test_missing_sdk_does_not_look_like_missing_engineering_data(self):
        with patch.object(ai.os, "getenv", return_value="synthetic-test-value"), \
             patch.dict(sys.modules, {"anthropic": None}), \
             self.assertRaises(ModelCallError) as raised:
            ai._client()
        self.assertEqual(raised.exception.code, "sdk_unavailable")
        self.assertEqual(raised.exception.request_state, "not_sent")


class ModelResponseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = patch.object(ai, "CACHE_DIR", Path(self.temp.name))
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.enable_cache = patch.object(ai, "USE_CACHE", True)
        self.enable_cache.start()
        self.addCleanup(self.enable_cache.stop)
        self.client_patch = patch.object(ai, "_client")
        self.client = self.client_patch.start().return_value
        self.addCleanup(self.client_patch.stop)
        self.client.messages.create.return_value = response()

    def test_uncached_request_reaches_messages_create_and_records_usage(self):
        spec, usage = call_model()
        self.assertEqual(spec.name, "Parsed bracket")
        self.assertFalse(usage.cache_hit)
        self.assertEqual((usage.input_tokens, usage.output_tokens), (12, 20))
        self.client.messages.create.assert_called_once()
        request = self.client.messages.create.call_args.kwargs
        self.assertEqual(request["model"], ai.MODEL)
        self.assertEqual(json.loads(request["messages"][0]["content"]), {"request": "brief"})

    def test_multiple_text_blocks_and_non_text_blocks_are_handled(self):
        self.client.messages.create.return_value = response(content=[
            SimpleNamespace(type="thinking", thinking="ignored reasoning"),
            SimpleNamespace(type="text", text='{"name":'),
            SimpleNamespace(type="text", text='"Multi-block bracket"}'),
        ])
        self.assertEqual(call_model()[0].name, "Multi-block bracket")

    def test_failure_categories_do_not_echo_bad_content_or_cache_it(self):
        cases = [
            (response(PRIVATE), "invalid_json"),
            (response('{"unexpected":"' + PRIVATE + '"}'), "invalid_schema"),
            (response(content=[]), "empty_response"),
            (response(content=None), "empty_response"),
            (response(content=[SimpleNamespace(type="tool_use")]), "empty_response"),
            (response(stop_reason="max_tokens"), "truncated_response"),
        ]
        for provider_response, code in cases:
            with self.subTest(code=code):
                self.client.messages.create.return_value = provider_response
                with self.assertRaises(ModelCallError) as raised:
                    call_model()
                self.assertEqual(raised.exception.code, code)
                self.assertEqual(raised.exception.request_state, "response_received")
                self.assertNotIn(PRIVATE, str(raised.exception))
                self.assertEqual(list(Path(self.temp.name).iterdir()), [])

    def test_sdk_error_reaches_ui_as_controlled_diagnostic(self):
        provider_error = RuntimeError(PRIVATE)
        provider_error.status_code = 401
        self.client.messages.create.side_effect = provider_error
        with self.assertRaises(ModelCallError) as raised:
            call_model()
        self.assertEqual(raised.exception.code, "authentication_failed")
        self.assertNotIn(PRIVATE, str(raised.exception))

    def test_valid_cache_does_not_claim_a_new_provider_call(self):
        first, first_usage = call_model()
        self.client.messages.create.reset_mock()
        second, cached_usage = call_model()
        self.assertEqual(first, second)
        self.assertFalse(first_usage.cache_hit)
        self.assertTrue(cached_usage.cache_hit)
        self.assertEqual(cached_usage.input_tokens, 0)
        self.client.messages.create.assert_not_called()

    def test_resolution_refresh_replaces_previously_cached_patch(self):
        spec = ProblemSpec(name="Original bracket")
        rejected = {"updates": [{"path": ["missing_field"], "value": "invalid edit"}]}
        valid = {"updates": [{"path": ["name"], "value": "Clarified bracket"}]}
        self.client.messages.create.side_effect = [response(json.dumps(rejected)), response(json.dumps(valid))]

        first, first_usage = ai.resolve_answers(spec, Review(), {"name": "Clarified bracket"})
        self.assertEqual(first.updates[0].path, ["missing_field"])
        self.assertFalse(first_usage.cache_hit)

        refreshed, refreshed_usage = ai.resolve_answers(
            spec, Review(), {"name": "Clarified bracket"}, bypass_cache=True,
        )
        self.assertEqual(refreshed.updates[0].path, ["name"])
        self.assertFalse(refreshed_usage.cache_hit)
        self.assertEqual(self.client.messages.create.call_count, 2)

        saved, saved_usage = ai.resolve_answers(spec, Review(), {"name": "Clarified bracket"})
        self.assertEqual(saved, refreshed)
        self.assertTrue(saved_usage.cache_hit)
        self.assertEqual(self.client.messages.create.call_count, 2)

    def test_corrupt_or_obsolete_cache_refetches(self):
        key = ai._cache_key("parse", "test instructions", {"request": "brief"})
        for content in ("{broken", "null", "[]", '{"response_text":42}',
                        '{"response_text":"no JSON"}',
                        json.dumps({"response_text": '{"obsolete_field":true}'})):
            with self.subTest(content=content):
                ai._cache_path(key).write_text(content, encoding="utf-8")
                self.client.messages.create.reset_mock()
                spec, usage = call_model()
                self.assertEqual(spec.name, "Parsed bracket")
                self.assertFalse(usage.cache_hit)
                self.client.messages.create.assert_called_once()

    def test_unwritable_cache_cannot_discard_successful_response(self):
        with patch.object(ai, "_write_cache", side_effect=PermissionError("read-only")):
            spec, usage = call_model()
        self.assertEqual(spec.name, "Parsed bracket")
        self.assertEqual(usage.output_tokens, 20)

    def test_disabled_cache_neither_reads_nor_writes(self):
        with patch.object(ai, "USE_CACHE", False), \
             patch.object(ai, "_read_cache") as read, \
             patch.object(ai, "_write_cache") as write:
            call_model()
        read.assert_not_called()
        write.assert_not_called()


if __name__ == "__main__":
    unittest.main()

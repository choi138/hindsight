"""Quota-exhausted embedding 429s defer the operation instead of burning its retries.

``generate_embeddings_batch`` used to re-raise every backend failure as a bare
``Exception`` without a cause, so an OpenAI ``insufficient_quota`` 429 (which no
retry can clear) went through the worker's generic fixed retry policy and was then
marked permanently failed. The LLM path already turns such responses into
``ProviderRateLimitResetError``, which the task runner converts into a defer that
does not consume ``retry_count``; the embedding path now does the same.

Also pins the OpenAI-SDK embedding client wiring: an explicit per-attempt timeout
(the SDK default is 600s) and ``max_retries`` from ``HINDSIGHT_API_EMBEDDINGS_*``.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import RateLimitError

from hindsight_api.engine.llm_interface import ProviderRateLimitResetError
from hindsight_api.engine.retain import embedding_utils
from hindsight_api.engine.retain.embedding_utils import EmbeddingBatchError, generate_embeddings_batch

OPENAI_ENV_VARS = [
    "HINDSIGHT_API_EMBEDDINGS_PROVIDER",
    "HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY",
    "HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL",
    "HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL",
    "HINDSIGHT_API_EMBEDDINGS_MAX_RETRIES",
    "HINDSIGHT_API_EMBEDDINGS_REQUEST_TIMEOUT",
    "HINDSIGHT_API_EMBEDDINGS_QUOTA_DEFER_SECONDS",
    "HINDSIGHT_API_LLM_API_KEY",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    from hindsight_api.config import clear_config_cache

    for env_var in OPENAI_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PROVIDER", "rrf")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_PROVIDER", "openai")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY", "sk-test")
    clear_config_cache()

    yield

    clear_config_cache()


def _rate_limit_error(message: str, *, headers: dict[str, str] | None = None, body: object = None) -> RateLimitError:
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    response = httpx.Response(429, headers=headers or {}, request=request)
    return RateLimitError(message, response=response, body=body)


def _insufficient_quota_error(**kwargs) -> RateLimitError:
    body = {
        "message": "You exceeded your current quota, please check your plan and billing details.",
        "type": "insufficient_quota",
        "code": "insufficient_quota",
    }
    return _rate_limit_error(f"Error code: 429 - {{'error': {body}}}", body=body, **kwargs)


def _backend(error: BaseException) -> MagicMock:
    backend = MagicMock()
    backend.dimension = 2
    backend.encode_documents = AsyncMock(side_effect=error)
    backend.encode_query = AsyncMock(side_effect=error)
    return backend


def test_insufficient_quota_429_defers_until_configured_window():
    err = _insufficient_quota_error()
    before = datetime.now(UTC)

    with pytest.raises(ProviderRateLimitResetError) as exc_info:
        asyncio.run(generate_embeddings_batch(_backend(err), ["a", "b"]))

    retry_at = exc_info.value.retry_at
    assert retry_at > before + timedelta(seconds=890)
    assert retry_at <= datetime.now(UTC) + timedelta(seconds=900)
    assert exc_info.value.__cause__ is err
    assert "insufficient_quota" in str(exc_info.value)


def test_quota_defer_window_is_configurable(monkeypatch):
    from hindsight_api.config import clear_config_cache

    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_QUOTA_DEFER_SECONDS", "120")
    clear_config_cache()
    before = datetime.now(UTC)

    with pytest.raises(ProviderRateLimitResetError) as exc_info:
        asyncio.run(generate_embeddings_batch(_backend(_insufficient_quota_error()), ["a"]))

    assert before + timedelta(seconds=110) < exc_info.value.retry_at <= datetime.now(UTC) + timedelta(seconds=120)


def test_retry_after_header_beats_the_configured_window():
    err = _insufficient_quota_error(headers={"Retry-After": "30"})
    before = datetime.now(UTC)

    with pytest.raises(ProviderRateLimitResetError) as exc_info:
        asyncio.run(generate_embeddings_batch(_backend(err), ["a"], input_type="query"))

    assert before + timedelta(seconds=25) < exc_info.value.retry_at <= datetime.now(UTC) + timedelta(seconds=30)


@pytest.mark.parametrize(
    "message",
    [
        "Error code: 429 - Your credit balance is too low to access the API.",
        "Error code: 429 - This request requires more credits. No credits remaining.",
        "Error code: 429 - Insufficient credits. Add more using https://openrouter.ai/settings/credits",
    ],
)
def test_credit_wording_counts_as_quota_exhaustion(message):
    with pytest.raises(ProviderRateLimitResetError):
        asyncio.run(generate_embeddings_batch(_backend(_rate_limit_error(message)), ["a"]))


@pytest.mark.parametrize(
    "message",
    [
        # OpenAI per-minute limit: mentions the billing page, but clears on its own.
        "Error code: 429 - Rate limit reached for requests. Please try again in 20s. "
        "You can increase your rate limit by adding a payment method at /account/billing.",
        # Gemini per-minute RESOURCE_EXHAUSTED reuses the exhausted-quota sentence.
        "429 RESOURCE_EXHAUSTED. You exceeded your current quota, please check your plan "
        "and billing details. Please retry in 32.5s.",
    ],
)
def test_burst_limit_429_is_not_deferred(message):
    """A burst limit clears on its own; it stays on the normal retry path."""
    err = _rate_limit_error(message)

    with pytest.raises(EmbeddingBatchError) as exc_info:
        asyncio.run(generate_embeddings_batch(_backend(err), ["a"]))

    assert exc_info.value.__cause__ is err


def test_status_less_rate_limit_error_is_not_deferred():
    class RateLimitError(Exception):
        pass

    with pytest.raises(EmbeddingBatchError):
        asyncio.run(generate_embeddings_batch(_backend(RateLimitError("insufficient_quota")), ["a"]))


def test_non_quota_error_keeps_wrapped_message_and_cause():
    err = TimeoutError("read timed out")

    with pytest.raises(EmbeddingBatchError) as exc_info:
        asyncio.run(generate_embeddings_batch(_backend(err), ["a"]))

    assert str(exc_info.value) == "Failed to generate batch embeddings: read timed out"
    assert exc_info.value.__cause__ is err
    assert isinstance(exc_info.value, Exception)


def test_backend_raised_reset_error_passes_through_untouched():
    retry_at = datetime.now(UTC) + timedelta(hours=1)
    err = ProviderRateLimitResetError(retry_at=retry_at, message="upstream says wait")

    with pytest.raises(ProviderRateLimitResetError) as exc_info:
        asyncio.run(generate_embeddings_batch(_backend(err), ["a"]))

    assert exc_info.value is err


def test_classifier_ignores_non_429_status_with_quota_wording():
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    err = httpx.HTTPStatusError("insufficient_quota", request=request, response=httpx.Response(400, request=request))
    assert embedding_utils._is_quota_exhausted_rate_limit(err) is False


@pytest.mark.asyncio
async def test_openai_client_is_built_with_timeout_and_max_retries():
    from hindsight_api.engine.embeddings import OpenAIEmbeddings

    emb = OpenAIEmbeddings(api_key="sk-test", model="text-embedding-3-small", max_retries=2, timeout=12.5)

    with patch("openai.AsyncOpenAI") as async_openai:
        await emb.initialize()
        emb._loop_client()

    async_openai.assert_called_once()
    kwargs = async_openai.call_args.kwargs
    assert kwargs["timeout"] == 12.5
    assert kwargs["max_retries"] == 2


def test_openai_client_timeout_defaults_to_config_default():
    from hindsight_api.config import DEFAULT_EMBEDDINGS_REQUEST_TIMEOUT
    from hindsight_api.engine.embeddings import OpenAIEmbeddings

    emb = OpenAIEmbeddings(api_key="sk-test")

    assert emb.timeout == DEFAULT_EMBEDDINGS_REQUEST_TIMEOUT == 60.0


def test_factory_wires_timeout_and_max_retries_from_config(monkeypatch):
    from hindsight_api.config import clear_config_cache
    from hindsight_api.engine.embeddings import OpenAIEmbeddings, create_embeddings_from_env

    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_MAX_RETRIES", "7")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_REQUEST_TIMEOUT", "21.5")
    clear_config_cache()

    emb = create_embeddings_from_env()

    assert isinstance(emb, OpenAIEmbeddings)
    assert emb.max_retries == 7
    assert emb.timeout == 21.5


def test_request_timeout_must_be_positive(monkeypatch):
    from hindsight_api.config import HindsightConfig

    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_REQUEST_TIMEOUT", "0")

    with pytest.raises(ValueError, match="HINDSIGHT_API_EMBEDDINGS_REQUEST_TIMEOUT must be > 0"):
        HindsightConfig.from_env()


def test_quota_defer_default_is_fifteen_minutes():
    from hindsight_api.config import HindsightConfig

    assert HindsightConfig.from_env().embeddings_quota_defer_seconds == 900.0


def test_quota_defer_window_must_be_positive(monkeypatch):
    """0 would re-queue the parked operation immediately, forever, without ever failing it."""
    from hindsight_api.config import HindsightConfig

    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_QUOTA_DEFER_SECONDS", "0")

    with pytest.raises(ValueError, match="HINDSIGHT_API_EMBEDDINGS_QUOTA_DEFER_SECONDS must be > 0"):
        HindsightConfig.from_env()

"""Consolidation must not starve user traffic or outlive ``enable_auto_consolidation``.

Three guards on the round loop:

* The per-fact recalls of an LLM batch are bounded by ``consolidation_recall_max_concurrent``.
  Unbounded, ``llm_batch_size x llm_parallelism`` recalls run at once, each holding up to
  ``recall_connection_budget`` connections — enough to exhaust a small DB pool and fail user
  recalls (and the consolidation batches themselves) with acquire timeouts.
* A round that hits ``consolidation_max_memories_per_round`` does not re-queue itself when
  ``enable_auto_consolidation`` is off, and the worker drops a queued automatic continuation
  the same way — so flipping the flag stops an in-flight chain, not only new triggers. An
  explicit (caller-requested) run keeps draining.
* ``consolidation_requeue_delay_seconds`` lands on the re-queued row as ``next_retry_at``,
  which the claim query honours, so the continuation is not claimable before the delay.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from hindsight_api.config import HindsightConfig, _get_raw_config, clear_config_cache
from hindsight_api.engine.consolidation.consolidator import (
    _ConsolidationBatchResponse,
    _CreateAction,
    run_consolidation_job,
)
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.providers.mock_llm import MockLLM
from hindsight_api.engine.response_models import RecallResult
from hindsight_api.engine.task_backend import WorkerTaskBackend


@pytest.fixture(autouse=True)
def enable_observations():
    config = _get_raw_config()
    original = config.enable_observations
    config.enable_observations = True
    yield
    config.enable_observations = original


def _override_config(memory: MemoryEngine, **overrides):
    raw = _get_raw_config()
    fake = type(raw)(**{**{f: getattr(raw, f) for f in raw.__dataclass_fields__}, **overrides})
    return patch.object(memory._config_resolver, "resolve_full_config", return_value=fake)


def _one_obs_per_fact_llm():
    mock_llm = MockLLM(provider="mock", api_key="", base_url="", model="mock-model")

    def callback(messages, scope):
        if scope != "consolidation":
            return _ConsolidationBatchResponse()
        prompt = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
        fact_ids = re.findall(r"\[([0-9a-f-]{36})\]", prompt)
        return _ConsolidationBatchResponse(
            creates=[_CreateAction(text=f"Observation about {fid[:8]}", source_fact_ids=[fid]) for fid in fact_ids]
        )

    mock_llm.set_response_callback(callback)
    wrapper = MagicMock()
    wrapper.with_config.return_value = mock_llm
    return wrapper


async def _insert_memory(conn, bank_id: str, text: str) -> None:
    await conn.execute(
        """
        INSERT INTO memory_units (id, bank_id, text, fact_type, tags, created_at)
        VALUES ($1, $2, $3, 'experience', $4, now())
        """,
        uuid.uuid4(),
        bank_id,
        text,
        [],
    )


async def _retain_backlog(memory: MemoryEngine, bank_id: str, request_context, n: int) -> None:
    with _override_config(memory, enable_observations=False):
        for i in range(n):
            await memory.retain_async(
                bank_id=bank_id,
                content=f"Fact {i}: the user did activity number {i} on day {i}.",
                request_context=request_context,
            )


async def _count_unconsolidated(memory: MemoryEngine, bank_id: str, request_context) -> int:
    page = await memory.list_memory_units(
        bank_id=bank_id, consolidation_state="pending", limit=1, request_context=request_context
    )
    return page["total"]


async def _pending_consolidation_rows(memory: MemoryEngine, bank_id: str) -> list:
    async with memory._pool.acquire() as conn:
        return await conn.fetch(
            """
            SELECT operation_id, task_payload, next_retry_at FROM async_operations
            WHERE bank_id = $1 AND operation_type = 'consolidation' AND status = 'pending'
            """,
            bank_id,
        )


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
@pytest.mark.parametrize("limit", [2, 8])
async def test_batch_recalls_are_bounded_by_consolidation_recall_max_concurrent(
    memory: MemoryEngine, request_context, limit: int
):
    """8 facts in one LLM batch fan out 8 recalls; at most ``limit`` are in flight at once."""
    bank_id = f"recall-cap-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        async with memory._pool.acquire() as conn:
            for i in range(8):
                await _insert_memory(conn, bank_id, f"The user likes activity number {i}.")

        in_flight = 0
        peak = 0
        calls = 0

        async def tracked_recall(*_args, **_kwargs):
            nonlocal in_flight, peak, calls
            calls += 1
            in_flight += 1
            peak = max(peak, in_flight)
            try:
                await asyncio.sleep(0.02)
            finally:
                in_flight -= 1
            return RecallResult(results=[])

        original_llm = memory._consolidation_llm_config
        original_sem = memory._consolidation_recall_semaphore
        memory._consolidation_llm_config = _one_obs_per_fact_llm()
        memory._consolidation_recall_semaphore = asyncio.Semaphore(limit)
        try:
            with (
                _override_config(memory, consolidation_llm_batch_size=8, consolidation_llm_parallelism=4),
                patch.object(memory, "recall_async", side_effect=tracked_recall),
                patch.object(memory, "submit_async_consolidation"),
            ):
                result = await run_consolidation_job(
                    memory_engine=memory, bank_id=bank_id, request_context=request_context
                )
        finally:
            memory._consolidation_llm_config = original_llm
            memory._consolidation_recall_semaphore = original_sem

        assert result["status"] == "completed"
        assert calls == 8
        # The unbounded case (limit 8) proves the stub observes the fan-out, so the
        # bounded assertion is not vacuous.
        assert peak == min(limit, 8)
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


def test_consolidation_recall_max_concurrent_is_read_from_env():
    saved = os.environ.get("HINDSIGHT_API_CONSOLIDATION_RECALL_MAX_CONCURRENT")
    saved_delay = os.environ.get("HINDSIGHT_API_CONSOLIDATION_REQUEUE_DELAY_SECONDS")
    os.environ["HINDSIGHT_API_CONSOLIDATION_RECALL_MAX_CONCURRENT"] = "2"
    os.environ["HINDSIGHT_API_CONSOLIDATION_REQUEUE_DELAY_SECONDS"] = "15"
    clear_config_cache()
    try:
        config = HindsightConfig.from_env()
        assert config.consolidation_recall_max_concurrent == 2
        assert config.consolidation_requeue_delay_seconds == 15.0
    finally:
        for key, value in (
            ("HINDSIGHT_API_CONSOLIDATION_RECALL_MAX_CONCURRENT", saved),
            ("HINDSIGHT_API_CONSOLIDATION_REQUEUE_DELAY_SECONDS", saved_delay),
        ):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        clear_config_cache()


@pytest.mark.asyncio
async def test_round_limit_does_not_requeue_when_auto_consolidation_disabled(memory: MemoryEngine, request_context):
    bank_id = f"auto-off-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        await _retain_backlog(memory, bank_id, request_context, 6)
        assert await _count_unconsolidated(memory, bank_id, request_context) >= 6

        with (
            _override_config(memory, consolidation_max_memories_per_round=3, enable_auto_consolidation=False),
            patch.object(memory, "submit_async_consolidation") as mock_requeue,
        ):
            result = await run_consolidation_job(memory_engine=memory, bank_id=bank_id, request_context=request_context)

        assert result["status"] == "completed"
        assert 0 < result["memories_processed"] <= 3
        mock_requeue.assert_not_called()
        assert await _count_unconsolidated(memory, bank_id, request_context) > 0
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_caller_requested_round_limit_requeues_despite_auto_consolidation_disabled(
    memory: MemoryEngine, request_context
):
    """With auto-consolidation off, consolidation runs only when explicitly triggered — and
    that explicit run must still drain the whole bank, so its chain carries the flag on."""
    bank_id = f"auto-off-explicit-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        await _retain_backlog(memory, bank_id, request_context, 6)

        with (
            _override_config(memory, consolidation_max_memories_per_round=3, enable_auto_consolidation=False),
            patch.object(memory, "submit_async_consolidation") as mock_requeue,
        ):
            await run_consolidation_job(
                memory_engine=memory, bank_id=bank_id, request_context=request_context, caller_requested=True
            )

        mock_requeue.assert_called_once()
        assert mock_requeue.call_args.kwargs["caller_requested"] is True
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_worker_drops_queued_automatic_consolidation_when_auto_disabled(memory: MemoryEngine, request_context):
    """A continuation queued before the flag flipped is skipped when claimed; an explicit one runs."""
    bank_id = f"auto-off-worker-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        await _retain_backlog(memory, bank_id, request_context, 2)
        before = await _count_unconsolidated(memory, bank_id, request_context)

        async def run_queued(extra_payload: dict) -> str:
            op_id = uuid.uuid4()
            payload = {"type": "consolidation", "operation_id": str(op_id), "bank_id": bank_id, **extra_payload}
            async with memory._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
                    VALUES ($1, $2, 'consolidation', 'processing', $3::jsonb)
                    """,
                    op_id,
                    bank_id,
                    json.dumps(payload),
                )
            with _override_config(memory, enable_auto_consolidation=False):
                await memory.execute_task(payload)
            async with memory._pool.acquire() as conn:
                return await conn.fetchval("SELECT status FROM async_operations WHERE operation_id = $1", op_id)

        assert await run_queued({}) == "completed"
        assert await _count_unconsolidated(memory, bank_id, request_context) == before, (
            "an automatic consolidation claimed after the flag flipped must not process anything"
        )

        assert await run_queued({"_caller_requested": True}) == "completed"
        assert await _count_unconsolidated(memory, bank_id, request_context) < before
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_requeue_delay_sets_next_retry_at_on_continuation(memory: MemoryEngine, request_context):
    bank_id = f"requeue-delay-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    original_backend = memory._task_backend
    # WorkerTaskBackend only inserts the row, as in production; SyncTaskBackend would run it.
    memory._task_backend = WorkerTaskBackend()
    await memory._task_backend.initialize()
    try:
        await _retain_backlog(memory, bank_id, request_context, 6)

        delay = 120.0
        with _override_config(
            memory, consolidation_max_memories_per_round=3, consolidation_requeue_delay_seconds=delay
        ):
            submitted_at = datetime.now(UTC)
            result = await run_consolidation_job(memory_engine=memory, bank_id=bank_id, request_context=request_context)

        assert result["status"] == "completed"
        rows = await _pending_consolidation_rows(memory, bank_id)
        assert len(rows) == 1
        next_retry_at = rows[0]["next_retry_at"]
        assert next_retry_at is not None, "the continuation must be parked on next_retry_at"
        assert (
            submitted_at + timedelta(seconds=delay - 5) <= next_retry_at <= submitted_at + timedelta(seconds=delay + 5)
        )
        payload = rows[0]["task_payload"]
        payload = json.loads(payload) if isinstance(payload, str) else payload
        assert "_caller_requested" not in payload
    finally:
        memory._task_backend = original_backend
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_requeue_without_delay_is_claimable_immediately(memory: MemoryEngine, request_context):
    bank_id = f"requeue-nodelay-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    original_backend = memory._task_backend
    memory._task_backend = WorkerTaskBackend()
    await memory._task_backend.initialize()
    try:
        await _retain_backlog(memory, bank_id, request_context, 6)
        with _override_config(memory, consolidation_max_memories_per_round=3, consolidation_requeue_delay_seconds=0):
            await run_consolidation_job(memory_engine=memory, bank_id=bank_id, request_context=request_context)

        rows = await _pending_consolidation_rows(memory, bank_id)
        assert len(rows) == 1
        assert rows[0]["next_retry_at"] is None
    finally:
        memory._task_backend = original_backend
        await memory.delete_bank(bank_id, request_context=request_context)

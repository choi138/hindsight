"""An accepted async retain can still end ``failed`` on the server (an embeddings
outage, say). The plugin used to treat that as stored and, since later append retains
only ship the delta, the turn was lost for good. Now the payload stays tracked until
the server reports ``completed`` and failed ops are re-sent with backoff."""

import asyncio
import logging
import time
from types import SimpleNamespace

import pytest

from conftest import FakeClient
from hindsight_client_api.exceptions import NotFoundException


class OpsClient(FakeClient):
    """Fake with the operations API. Each op's status follows ``outcome``:
    ``completed`` (stored first time), ``failed_until_retried`` (fails until the retry
    endpoint re-queues it), ``failed`` (never recovers) or ``pending`` (never finishes).
    Ops created by a re-sent payload complete, as a fresh submission normally would."""

    def __init__(self, outcome: str = "completed"):
        super().__init__()
        self.outcome = outcome
        self.attempted: list[dict] = []
        self.retries: list[str] = []
        self.retry_errors: list[Exception] = []
        self.fail_sends = 0
        self.on_retry = lambda operation_id: None
        self.operations = SimpleNamespace(get_operation_status=self._status, retry_operation=self._retry)

    async def aretain_batch(self, **kwargs):
        self.attempted.append(kwargs)
        if self.fail_sends:
            self.fail_sends -= 1
            raise ConnectionError("boom")
        self.retains.append(kwargs)
        return SimpleNamespace(operation_id=kwargs["operation_id"])

    async def _status(self, bank_id, operation_id):
        resent = any(r["operation_id"] == operation_id for r in self.retains[1:])
        if self.outcome == "completed" or resent or (operation_id in self.retries and self.outcome != "failed"):
            status = "completed"
        else:
            status = "pending" if self.outcome == "pending" else "failed"
        return SimpleNamespace(status=status)

    async def _retry(self, bank_id, operation_id):
        self.on_retry(operation_id)
        self.retries.append(operation_id)
        if self.retry_errors:
            raise self.retry_errors.pop(0)
        return SimpleNamespace(operation_id=operation_id)


def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.02)


@pytest.fixture
def tracked(provider, monkeypatch):
    """A provider on an OpsClient with the background timers shrunk to test speed."""

    def _make(outcome: str = "completed", config: dict | None = None):
        fake = OpsClient(outcome)
        instance, _ = provider(config or {}, client=fake)
        monkeypatch.setattr(instance, "_WRITER_IDLE_TICK_S", 0.02)
        monkeypatch.setattr(instance, "_RETAIN_OP_BACKGROUND_POLL_S", (0.02, 0.02))
        monkeypatch.setattr(instance, "_RETAIN_REQUEUE_BACKOFF_S", (0.0, 0.0))
        return instance, fake

    return _make


def test_a_failed_op_is_retried_server_side_and_its_payload_kept_until_it_completes(tracked):
    instance, fake = tracked("failed_until_retried")
    held_at_retry = []
    fake.on_retry = lambda op: held_at_retry.append(
        [(r.document_id, r.resend is not None) for r in instance._tracked_retains]
    )

    instance.sync_turn("what is my name?", "Ada.")
    _wait_until(lambda: len(fake.retains) == 1)
    op_id = fake.retains[0]["operation_id"]

    _wait_until(lambda: fake.retries == [op_id])
    _wait_until(lambda: instance._tracked_retains == [])
    instance.shutdown()

    # The payload was still held, re-sendable, when the server was asked to retry ...
    assert held_at_retry == [[("session-1", True)]]
    # ... and the server did the work: the content went over the wire once.
    assert len(fake.retains) == 1


def test_the_payload_is_sent_again_when_the_failed_op_is_gone_from_the_server(tracked):
    instance, fake = tracked("failed_until_retried")
    fake.retry_errors = [NotFoundException(status=404, reason="Not Found")]

    instance.sync_turn("what is my name?", "Ada.")
    _wait_until(lambda: len(fake.retains) == 2)
    _wait_until(lambda: instance._tracked_retains == [])
    instance.shutdown()

    first, second = fake.retains
    assert second["items"] == first["items"] and second["document_id"] == first["document_id"]
    # A replay of the known id would just return the failed op, so the re-send is a new op.
    assert second["operation_id"] != first["operation_id"]
    assert fake.retries == [first["operation_id"]]


def test_a_send_that_raised_is_re_sent_under_the_same_operation_id(tracked):
    """A lost acknowledgement must not append the turn twice: the server replays a known id."""
    instance, fake = tracked()
    fake.fail_sends = 1

    instance.sync_turn("what is my name?", "Ada.")
    _wait_until(lambda: len(fake.retains) == 1)
    _wait_until(lambda: instance._tracked_retains == [])
    instance.shutdown()

    assert len(fake.attempted) == 2
    assert fake.attempted[0]["operation_id"] == fake.retains[0]["operation_id"]


def test_after_the_last_attempt_the_plugin_logs_the_document_and_stops(tracked, caplog):
    instance, fake = tracked("failed", {"retain_max_requeue_attempts": 1})

    with caplog.at_level(logging.ERROR, logger="hindsight_hermes"):
        instance.sync_turn("what is my name?", "Ada.")
        _wait_until(lambda: len(fake.retains) == 1)
        _wait_until(lambda: instance._tracked_retains == [])
        time.sleep(0.2)
        instance.shutdown()

    op_id = fake.retains[0]["operation_id"]
    assert fake.retries == [op_id]
    assert len(fake.retains) == 1
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "document session-1" in errors[0] and f"/operations/{op_id}/retry" in errors[0]


def test_the_prefetch_stops_waiting_at_its_deadline_but_the_op_stays_tracked(tracked):
    instance, fake = tracked("pending")
    instance.sync_turn("what is my name?", "Ada.")
    _wait_until(lambda: len(fake.retains) == 1)

    assert instance._wait_for_retains_drained(0.3) is False
    (record,) = instance._tracked_retains
    assert record.gated is False and record.state == "pending"
    # Later prefetches no longer pay for it ...
    started = time.monotonic()
    assert instance._wait_for_retains_drained(0.3) is True
    assert time.monotonic() - started < 0.2
    # ... but it is still polled, so a later failure is still re-sent.
    fake.outcome = "failed_until_retried"
    _wait_until(lambda: instance._tracked_retains == [])
    assert fake.retries == [fake.retains[0]["operation_id"]]
    instance.shutdown()


def test_a_slow_status_poll_is_cut_off_at_its_own_timeout(tracked, monkeypatch):
    instance, fake = tracked()

    async def slow(bank_id, operation_id):
        await asyncio.sleep(5)
        return SimpleNamespace(status="completed")

    fake.operations.get_operation_status = slow
    monkeypatch.setattr(instance, "_RETAIN_OP_STATUS_TIMEOUT_S", 0.2)

    started = time.monotonic()
    assert instance._retain_op_status("hermes", "op-1") is None
    assert time.monotonic() - started < 2.0

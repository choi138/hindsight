"""A retain that fails mid-extraction must leave the document as it was.

The streaming retain extracts every chunk concurrently and only commits the first consumer
batch once extraction is done. That first commit is what REPLACES the document: it deletes
the prior memory units and overwrites ``original_text``. Chunks that extract zero facts never
reach the embedder, so during an embeddings outage they "succeed" while every fact-bearing
chunk fails -- and the consumer used to commit the 0-fact survivors (document tracking
included) before the orchestrator raised the extraction error. Production ended up with
documents whose previous memories were gone, whose text was the new version, and which had
no memories at all (883 of them after one quota outage).

The counterpart must keep working: a document that genuinely extracts nothing, with no error,
is still recorded (and a re-ingest that now yields nothing still drops the outgoing memories).
"""

from datetime import datetime, timezone
from typing import Any

import pytest

from hindsight_api.config import clear_config_cache
from hindsight_api.engine.response_models import TokenUsage
from hindsight_api.engine.retain import fact_extraction
from hindsight_api.engine.retain.types import ChunkMetadata, ExtractedFact, ExtractionResult, RetainContent

pytestmark = pytest.mark.memory_backend_incompatible

FACT_MARKER = "FACT_BEARING"


@pytest.fixture(autouse=True)
def _retain_only_env(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_AUTO_CONSOLIDATION", "false")
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_OBSERVATIONS", "false")
    clear_config_cache()
    yield
    clear_config_cache()


@pytest.fixture
def extraction_calls(monkeypatch) -> list[str]:
    """Stub extraction: one fact per content that carries ``FACT_MARKER``, none otherwise.

    Records each extracted content so a test can prove the document was split into the chunks
    it meant to exercise.
    """
    calls: list[str] = []

    async def _extract(contents: list[RetainContent], *args: Any, **kwargs: Any) -> ExtractionResult:
        facts: list[ExtractedFact] = []
        chunks: list[ChunkMetadata] = []
        for index, item in enumerate(contents):
            calls.append(item.content)
            bearing = FACT_MARKER in item.content
            chunks.append(
                ChunkMetadata(
                    chunk_text=item.content,
                    fact_count=1 if bearing else 0,
                    content_index=index,
                    chunk_index=index,
                )
            )
            if bearing:
                facts.append(
                    ExtractedFact(
                        fact_text=f"Synthetic fact from: {item.content[:40]}",
                        fact_type="world",
                        content_index=index,
                        chunk_index=index,
                        context=item.context,
                        tags=item.tags,
                    )
                )
        return ExtractionResult(facts, chunks, TokenUsage())

    monkeypatch.setattr(fact_extraction, "extract_facts_from_contents", _extract)
    return calls


class EmbeddingsOutage(RuntimeError):
    pass


@pytest.fixture
def embeddings_switch(monkeypatch, memory):
    """Replace the embeddings backend with one a test can flip into an outage."""
    state = {"failing": False}

    async def _encode_documents(texts: list[str]) -> list[list[float]]:
        if state["failing"]:
            raise EmbeddingsOutage("embeddings quota exceeded")
        return [[0.0] * memory.embeddings.dimension for _ in texts]

    monkeypatch.setattr(memory.embeddings, "encode_documents", _encode_documents)
    return state


def _paragraph(label: str, *, chars: int) -> str:
    words = f"{label} alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima. "
    return (words * (chars // len(words) + 1))[:chars]


def _two_chunk_body(prefix: str) -> str:
    # Plain text over the default 3000-char chunk size with one paragraph break: the chunker
    # cuts it there, giving a fact-bearing chunk and a 0-fact chunk in a single document.
    return _paragraph(f"{prefix} {FACT_MARKER}", chars=1800) + "\n\n" + _paragraph(f"{prefix} quiet", chars=1800)


@pytest.mark.asyncio
async def test_failed_extraction_does_not_replace_the_document(
    memory, request_context, extraction_calls, embeddings_switch
):
    bank_id = f"test_retain_failed_keeps_doc_{datetime.now(timezone.utc).timestamp()}"
    document_id = "doc-outage"
    first_body = f"First version. {FACT_MARKER} Alice moved to Berlin and joined Acme."
    second_body = _two_chunk_body("Second version.")

    try:
        await memory.retain_async(
            bank_id=bank_id, content=first_body, document_id=document_id, request_context=request_context
        )
        before = await memory.get_document(document_id, bank_id, request_context=request_context)
        assert before is not None
        assert before["memory_unit_count"] == 1
        assert before["original_text"] == first_body

        extraction_calls.clear()
        embeddings_switch["failing"] = True
        with pytest.raises(Exception, match="embeddings quota exceeded"):
            await memory.retain_async(
                bank_id=bank_id, content=second_body, document_id=document_id, request_context=request_context
            )
        assert len(extraction_calls) == 2, "the second body must have been split into two chunks"
        assert sum(FACT_MARKER in c for c in extraction_calls) == 1

        after = await memory.get_document(document_id, bank_id, request_context=request_context)
        assert after is not None
        assert after["memory_unit_count"] == before["memory_unit_count"], (
            "the failed retain deleted the document's previous memory units"
        )
        assert after["original_text"] == first_body, "the failed retain overwrote the document's text"
        assert after["content_hash"] == before["content_hash"]

        # Once embeddings are back, re-sending the same body replaces the document normally.
        embeddings_switch["failing"] = False
        await memory.retain_async(
            bank_id=bank_id, content=second_body, document_id=document_id, request_context=request_context
        )
        recovered = await memory.get_document(document_id, bank_id, request_context=request_context)
        assert recovered is not None
        assert recovered["memory_unit_count"] == 1
        assert recovered["original_text"] == second_body
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_document_with_no_facts_and_no_error_is_still_recorded(
    memory, request_context, extraction_calls, embeddings_switch
):
    bank_id = f"test_retain_empty_doc_recorded_{datetime.now(timezone.utc).timestamp()}"
    try:
        empty_body = _paragraph("Nothing to extract here.", chars=200)
        unit_ids = await memory.retain_async(
            bank_id=bank_id, content=empty_body, document_id="doc-empty", request_context=request_context
        )
        assert unit_ids == []
        doc = await memory.get_document("doc-empty", bank_id, request_context=request_context)
        assert doc is not None
        assert doc["memory_unit_count"] == 0
        assert doc["original_text"] == empty_body

        # A re-ingest that genuinely yields nothing still replaces the previous version.
        await memory.retain_async(
            bank_id=bank_id,
            content=f"Has facts. {FACT_MARKER} Bob works at Globex.",
            document_id="doc-emptied",
            request_context=request_context,
        )
        assert (await memory.get_document("doc-emptied", bank_id, request_context=request_context))[
            "memory_unit_count"
        ] == 1
        emptied_body = _paragraph("Rewritten with nothing to extract.", chars=200)
        await memory.retain_async(
            bank_id=bank_id, content=emptied_body, document_id="doc-emptied", request_context=request_context
        )
        emptied = await memory.get_document("doc-emptied", bank_id, request_context=request_context)
        assert emptied is not None
        assert emptied["memory_unit_count"] == 0
        assert emptied["original_text"] == emptied_body
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)

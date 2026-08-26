"""Streaming vector retrieval layer over ChromaDB.

Each job gets its own persistent collection. Chunks are pushed in as the
transcript streams out of Whisper — `StreamingIndex.add()` buffers until a full
batch is ready, then embeds that batch in parallel against Ollama and upserts.
By the time transcription ends the index is already warm.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, Sequence

import config

_client = None
_client_lock = threading.Lock()


def get_client():
    """Process-wide persistent Chroma client (creating one per job is expensive)."""
    global _client
    with _client_lock:
        if _client is None:
            import chromadb
            from chromadb.config import Settings

            _client = chromadb.PersistentClient(
                path=str(config.CHROMA_DIR),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
    return _client


def _make_embedding_function():
    from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
    from langchain_ollama import OllamaEmbeddings

    class OllamaEmbeddingFunction(EmbeddingFunction):
        """Adapts LangChain's OllamaEmbeddings to Chroma's EF protocol."""

        def __init__(self) -> None:
            self._embedder = OllamaEmbeddings(
                model=config.OLLAMA_EMBED_MODEL,
                base_url=config.OLLAMA_BASE_URL,
            )
            self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="embed")

        def __call__(self, input: Documents) -> Embeddings:  # noqa: A002 - Chroma's API
            texts = list(input)
            if not texts:
                return []
            return list(self._pool.map(self._embedder.embed_query, texts))

        @staticmethod
        def name() -> str:
            return "ollama-clipfarm"

    return OllamaEmbeddingFunction()


class StreamingIndex:
    """Write-through buffer in front of a per-job Chroma collection."""

    def __init__(self, job_id: str, batch_size: int | None = None) -> None:
        self.job_id = job_id
        self.collection_name = f"job_{job_id}"
        self.batch_size = batch_size or config.EMBED_BATCH_SIZE
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._count = 0

        client = get_client()
        # Start clean so a re-run of the same job id never mixes in stale vectors.
        try:
            client.delete_collection(self.collection_name)
        except Exception:  # noqa: BLE001 - absent collection is the normal case
            pass
        self.collection = client.create_collection(
            name=self.collection_name,
            embedding_function=_make_embedding_function(),
            metadata={"hnsw:space": "cosine", "job_id": job_id},
        )

    @property
    def count(self) -> int:
        return self._count

    def add(self, chunk: dict[str, Any]) -> int:
        """Queue a chunk; flushes automatically once a batch is full.

        Returns the number of chunks written to Chroma by this call (0 or batch).
        """
        with self._lock:
            self._buffer.append(chunk)
            if len(self._buffer) < self.batch_size:
                return 0
            batch, self._buffer = self._buffer, []
        return self._write(batch)

    def add_many(self, chunks: Iterable[dict[str, Any]]) -> int:
        return sum(self.add(c) for c in chunks)

    def flush(self) -> int:
        """Write whatever is left in the buffer."""
        with self._lock:
            batch, self._buffer = self._buffer, []
        return self._write(batch)

    def _write(self, batch: Sequence[dict[str, Any]]) -> int:
        if not batch:
            return 0
        self.collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[_metadata(c) for c in batch],
        )
        self._count += len(batch)
        return len(batch)

    # --- retrieval ---

    def query(self, text: str, k: int = 8) -> list[dict[str, Any]]:
        """Nearest-neighbour search. Returns hits with a 0..1 `relevance`."""
        if self._count == 0:
            return []
        res = self.collection.query(
            query_texts=[text],
            n_results=min(k, self._count),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[dict[str, Any]] = []
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            hits.append({
                "id": cid,
                "text": doc,
                "metadata": dict(meta or {}),
                "distance": float(dist),
                # Chroma cosine distance is in [0, 2]; fold it back to a similarity.
                "relevance": round(max(0.0, 1.0 - float(dist)), 4),
            })
        return hits

    def multi_query(self, queries: Sequence[str], k: int = 8) -> dict[str, dict[str, Any]]:
        """Run several probe queries and merge hits per chunk id.

        A chunk matched by many different probes is a stronger candidate than one
        that only matches a single narrow probe, so we keep both the best
        relevance and the set of probes that surfaced it.
        """
        merged: dict[str, dict[str, Any]] = {}
        for q in queries:
            for hit in self.query(q, k=k):
                slot = merged.setdefault(hit["id"], {**hit, "matched_queries": []})
                slot["matched_queries"].append({"query": q, "relevance": hit["relevance"]})
                if hit["relevance"] > slot["relevance"]:
                    slot["relevance"] = hit["relevance"]
                    slot["distance"] = hit["distance"]
        for slot in merged.values():
            slot["match_count"] = len(slot["matched_queries"])
        return merged

    def drop(self) -> None:
        try:
            get_client().delete_collection(self.collection_name)
        except Exception:  # noqa: BLE001 - best effort cleanup
            pass


def open_index(job_id: str) -> Any:
    """Reopen an existing job collection for ad-hoc search after the job is done."""
    client = get_client()
    return client.get_collection(
        name=f"job_{job_id}",
        embedding_function=_make_embedding_function(),
    )


def search_job(job_id: str, text: str, k: int = 8) -> list[dict[str, Any]]:
    """Free-text semantic search across a finished job's transcript."""
    collection = open_index(job_id)
    total = collection.count()
    if total == 0:
        return []
    res = collection.query(
        query_texts=[text],
        n_results=min(k, total),
        include=["documents", "metadatas", "distances"],
    )
    out: list[dict[str, Any]] = []
    for cid, doc, meta, dist in zip(
        (res.get("ids") or [[]])[0],
        (res.get("documents") or [[]])[0],
        (res.get("metadatas") or [[]])[0],
        (res.get("distances") or [[]])[0],
    ):
        out.append({
            "id": cid,
            "text": doc,
            "start": float((meta or {}).get("start", 0.0)),
            "end": float((meta or {}).get("end", 0.0)),
            "relevance": round(max(0.0, 1.0 - float(dist)), 4),
        })
    return out


def _metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """Chroma metadata values must be scalars."""
    return {
        "start": float(chunk["start"]),
        "end": float(chunk["end"]),
        "duration": float(chunk["end"] - chunk["start"]),
        "n_words": int(chunk.get("n_words", 0)),
        "index": int(chunk.get("index", 0)),
    }

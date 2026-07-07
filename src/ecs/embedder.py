"""Embedding wrapper over sentence-transformers for the harrier-oss models.

Model: microsoft/harrier-oss-v1-0.6b (1024-dim, last-token pooling,
L2-normalized). Fallback: microsoft/harrier-oss-v1-270m via
ECS_EMBED_MODEL env var (same interface; note the 270m model may emit a
different native dim — check `Embedder.dim`).

- `embed_docs(texts)`: raw text, no prefix (title + abstract).
- `embed_query(q)` / `embed_queries(qs)`: prefixed with the instruction from
  ecs.config (`Instruct: {instruction}\nQuery: {q}`).

fp16 on CUDA when available, fp32 on CPU. Batched with a conservative
default batch size for a 12GB card; override via ECS_EMBED_BATCH.
"""

from __future__ import annotations

import os

import numpy as np

from . import config


def _query_prefix(q: str) -> str:
    return f"Instruct: {config.QUERY_INSTRUCTION}\nQuery: {q}"


class Embedder:
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        max_seq_length: int = 512,
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or config.EMBED_MODEL
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size or int(os.environ.get("ECS_EMBED_BATCH", "32"))
        kwargs = {}
        if device.startswith("cuda"):
            kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
        self.model = SentenceTransformer(
            self.model_name, device=device, trust_remote_code=True, **kwargs
        )
        self.model.max_seq_length = max_seq_length
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_docs(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed documents raw (no instruction prefix). Returns (n, dim) float32."""
        return self._encode(texts, show_progress)

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        """Embed queries with the retrieval instruction prefix."""
        return self._encode([_query_prefix(q) for q in queries], False)

    def embed_query(self, q: str) -> np.ndarray:
        """Embed a single query; returns a (dim,) float32 vector."""
        return self.embed_queries([q])[0]

    def _encode(self, texts: list[str], show_progress: bool) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,  # L2-normalized per DESIGN.md
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(vecs, dtype=np.float32)


_default: Embedder | None = None


def get_embedder() -> Embedder:
    """Process-wide lazily-constructed default Embedder (uses ECS_EMBED_MODEL)."""
    global _default
    if _default is None:
        _default = Embedder()
    return _default

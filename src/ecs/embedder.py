"""Embedding wrapper over sentence-transformers for the harrier-oss models.

Model: microsoft/harrier-oss-v1-270m (640-dim, Gemma3-based, last-token
pooling, L2-normalized) — the canonical embedder per DESIGN.md. Upgrade path:
microsoft/harrier-oss-v1-0.6b (1024-dim, Qwen3-based) via ECS_EMBED_MODEL
(same interface, different native dim — check `Embedder.dim`; requires a full
re-embed of the collection).

- `embed_docs(texts)`: raw text, no prefix (title + abstract).
- `embed_query(q)` / `embed_queries(qs)`: prefixed with the instruction from
  ecs.config (`Instruct: {instruction}\nQuery: {q}`).

Dtype on CUDA: bfloat16 when the GPU supports it, else float32 — NOT float16:
Gemma-based models overflow to NaN in fp16 (empirically 100% of rows for the
270m). Override with ECS_EMBED_DTYPE=float16|bfloat16|float32 if you know
better. CPU stays float32. Batched with a conservative default batch size for
a 12GB card; override via ECS_EMBED_BATCH.
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
            dtype_name = os.environ.get("ECS_EMBED_DTYPE")
            if dtype_name:
                dtype = getattr(torch, dtype_name)
            elif torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16  # fp16 NaNs on Gemma-based models
            else:
                dtype = torch.float32
            kwargs["model_kwargs"] = {"torch_dtype": dtype}
        self.model = SentenceTransformer(
            self.model_name, device=device, trust_remote_code=True, **kwargs
        )
        self.model.max_seq_length = max_seq_length
        get_dim = getattr(
            self.model,
            "get_embedding_dimension",  # sentence-transformers >= 5.x
            self.model.get_sentence_embedding_dimension,
        )
        self.dim = get_dim()

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
        out = np.asarray(vecs, dtype=np.float32)
        if not np.isfinite(out).all():
            bad = int((~np.isfinite(out)).any(axis=1).sum())
            raise ValueError(
                f"{bad}/{len(out)} embeddings contain NaN/inf "
                f"(model={self.model_name}, dtype overflow? see ECS_EMBED_DTYPE)"
            )
        return out


_default: Embedder | None = None


def get_embedder() -> Embedder:
    """Process-wide lazily-constructed default Embedder (uses ECS_EMBED_MODEL)."""
    global _default
    if _default is None:
        _default = Embedder()
    return _default

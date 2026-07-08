"""Side-by-side demo server: our graph-RL agent harness vs Exa.

Minimal FastAPI app (port 7900) with three routes:
  * GET  /       -> serves demo/index.html
  * POST /ask    -> runs eval.agent_baseline.AgentBaseline over the tool
                    server, then resolves each returned QID to
                    {qid, title, snippet} via the read-only sidecar sqlite.
  * POST /exa    -> proxies the question to Exa's /answer endpoint (falls
                    back to /search). Missing EXA_API_KEY -> graceful JSON.

Run:  .venv/bin/python -m uvicorn demo.server:app --port 7900
(from the repo root, so that `ecs`, `eval`, `questiongen` import cleanly and
ecs.config auto-loads the repo-root .env).
"""

from __future__ import annotations

import os
import sqlite3
import time

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from ecs import config  # auto-loads repo-root .env (OPENROUTER_API_KEY, EXA_API_KEY)
from eval.agent_baseline import AgentBaseline
from questiongen.schema import QuestionRecord

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML = os.path.join(HERE, "index.html")
SIDECAR_DB = os.path.join(config.DATA_DIR, "sidecar.db")

DEFAULT_MODEL = "minimax/minimax-m3"  # our best frontier number
TIMEOUT_S = 120.0

app = FastAPI(title="entity_component_search demo")


# --------------------------------------------------------------------------- #
# QID -> {title, snippet} resolution via the sidecar sqlite (read-only).
# --------------------------------------------------------------------------- #
def _sidecar_conn() -> sqlite3.Connection:
    """A fresh read-only connection (cheap; safe across request threads)."""
    return sqlite3.connect(f"file:{SIDECAR_DB}?mode=ro", uri=True)


def _snippet(abstract: str | None, limit: int = 200) -> str:
    if not abstract:
        return ""
    text = " ".join(abstract.split())
    # Prefer the first sentence if it lands within the limit, else hard-trim.
    dot = text.find(". ")
    if 0 < dot + 1 <= limit:
        return text[: dot + 1]
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def resolve_qids(qids: list[str]) -> list[dict]:
    """Map each QID to {qid, title, snippet}, preserving rank order."""
    if not qids:
        return []
    conn = _sidecar_conn()
    try:
        placeholders = ",".join("?" for _ in qids)
        rows = conn.execute(
            f"SELECT qid, title, abstract FROM entity WHERE qid IN ({placeholders})",
            qids,
        ).fetchall()
    finally:
        conn.close()
    by_qid = {qid: (title, abstract) for qid, title, abstract in rows}
    out: list[dict] = []
    for qid in qids:
        title, abstract = by_qid.get(qid, (None, None))
        out.append(
            {
                "qid": qid,
                "title": title or qid,  # bare QID if unknown to the sidecar
                "snippet": _snippet(abstract),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.post("/ask")
def ask(payload: dict) -> JSONResponse:
    """Run our agent harness, then resolve QIDs to titles + snippets."""
    question = (payload or {}).get("question", "").strip()
    model = (payload or {}).get("model") or DEFAULT_MODEL
    if not question:
        return JSONResponse({"error": "question is required"})

    t0 = time.time()
    try:
        if model.startswith("local:"):
            # trained checkpoint served on sglang (via SSH tunnel), OpenAI-compatible
            from openai import OpenAI as _OpenAI
            baseline = AgentBaseline(model=model.split(":", 1)[1])
            baseline._client = _OpenAI(
                base_url=os.environ.get("ECS_LOCAL_LLM_URL", "http://localhost:30010/v1"),
                api_key="EMPTY",
            )
        else:
            baseline = AgentBaseline(model=model)
            reason = baseline.unavailable_reason()
            if reason:
                return JSONResponse({"error": reason})
        record = QuestionRecord(
            id="demo",
            question=question,
            answer_qids=["Q1"],  # placeholder; scoring is not used here
            source="kg_pattern",
            difficulty=1,
        )
        qids = baseline.predict(record)
        entities = resolve_qids(qids)
        return JSONResponse(
            {
                "entities": entities,
                "raw_qids": qids,
                "seconds": round(time.time() - t0, 1),
                "model": model,
            }
        )
    except Exception as exc:  # errors as JSON, never a 500
        return JSONResponse(
            {"error": f"{type(exc).__name__}: {exc}", "seconds": round(time.time() - t0, 1)}
        )


@app.post("/exa")
def exa(payload: dict) -> JSONResponse:
    """Proxy the question to Exa (/answer, fallback /search)."""
    import httpx

    question = (payload or {}).get("question", "").strip()
    if not question:
        return JSONResponse({"error": "question is required"})

    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        return JSONResponse({"error": "EXA_API_KEY not set — add it to .env"})

    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    t0 = time.time()
    try:
        with httpx.Client(timeout=TIMEOUT_S) as client:
            # Preferred: Exa's answer endpoint (LLM answer + citations).
            r = client.post(
                "https://api.exa.ai/answer",
                headers=headers,
                json={"query": question, "text": True},
            )
            if r.status_code == 200:
                data = r.json()
                citations = data.get("citations") or []
                results = [
                    {
                        "title": c.get("title") or c.get("url") or "",
                        "url": c.get("url", ""),
                        "snippet": _snippet(c.get("text") or c.get("snippet")),
                    }
                    for c in citations
                ]
                return JSONResponse(
                    {
                        "answer": data.get("answer", ""),
                        "results": results,
                        "seconds": round(time.time() - t0, 1),
                    }
                )

            # Fallback: plain search with text contents.
            r = client.post(
                "https://api.exa.ai/search",
                headers=headers,
                json={"query": question, "numResults": 10, "contents": {"text": True}},
            )
            r.raise_for_status()
            data = r.json()
            results = [
                {
                    "title": res.get("title") or res.get("url") or "",
                    "url": res.get("url", ""),
                    "snippet": _snippet(res.get("text") or res.get("snippet")),
                }
                for res in (data.get("results") or [])
            ]
            return JSONResponse(
                {"results": results, "seconds": round(time.time() - t0, 1)}
            )
    except Exception as exc:  # errors as JSON, never a 500
        return JSONResponse(
            {"error": f"{type(exc).__name__}: {exc}", "seconds": round(time.time() - t0, 1)}
        )

# entity_component_search — Design Contract

SID-1-style agentic entity retrieval: train a small model (validation: Qwen3-0.6B, target: Qwen3-4B)
with GRPO to answer "report the entities needed to answer the question, ordered by relevance"
over a Wikipedia/Wikidata corpus, using vector + graph tools. Reward = two-tier entity-NDCG.

Reference: https://www.sid.ai/research/sid-1-technical-report (GRPO w/o SFT, NDCG reward,
TI/TO token handling, length scheduling, format reward).

## Fixed decisions (do not re-litigate; change only via this file)

- **Corpus**: Wikidata5M (transductive split): ~4.8M entities aligned with Wikipedia abstracts,
  ~21M triples, ~825 relations. Files: `wikidata5m_transductive_train.txt` (triples, TSV Q P Q),
  `wikidata5m_text.txt` (QID \t abstract), `wikidata5m_alias.tar.gz` (entity + relation aliases).
- **Graph DB**: Neo4j 5.x community (docker, bolt://localhost:7687, HTTP :7474, auth neo4j/ecs-local-dev).
  Nodes `(:Entity {qid, title})`, relationships typed by P-id (e.g. `[:P69]`) with `label` property
  holding the English relation name. Bulk load via `neo4j-admin database import full` from CSVs.
- **Vector DB**: Qdrant (docker, :6333 HTTP / :6334 gRPC). Collection `wiki_entities`,
  1024-dim, cosine, on-disk vectors + HNSW; payload `{qid, title}` (abstract text looked up
  from a local sqlite/parquet sidecar, NOT stored in Qdrant payload — keeps the index small).
- **Embedder**: `microsoft/harrier-oss-v1-0.6b` (1024-dim, last-token pooling, L2-normalized).
  Documents embedded raw (title + abstract). Queries embedded with:
  `Instruct: Given a question, retrieve Wikipedia entities relevant to answering it\nQuery: {q}`.
  `microsoft/harrier-oss-v1-270m` is the local/validation fallback (same interface).
- **Tool server**: FastAPI on :7801, JSON. Endpoints (also exposed as an OpenAI-style tool schema
  in `toolserver/schema.py`):
  - `POST /vector_search {query, k<=50}` → `[{qid, title, score, snippet}]`
  - `POST /get_entity {qid}` → `{qid, title, abstract, aliases, degree_in, degree_out}`
  - `POST /get_neighbors {qid, relation?, direction: out|in|both, limit<=100, offset}` →
    `{total, edges: [{src, rel, rel_label, dst, dst_title}]}` (paginated; hub-safe)
  - `POST /find_paths {src_qid, dst_qid, max_hops<=4, limit<=20}` → paths
- **Answer format** (the trained model's final output): a fenced block
  ```entities
  Q123  # optional comment
  Q456
  ```
  ordered by relevance, max 50 lines. Parser in `src/ecs/answer.py`.
- **Reward / metric**: two-tier NDCG. Gain 2 = answer-set entity, gain 1 = bridge/evidence
  entity, 0 = other. Ideal DCG computed from the golden set; list truncated at 50;
  malformed output → format penalty. Implementation lives in `eval/metrics.py` and is imported
  by training — single source of truth.
- **Question generation**, two pipelines, both emitting the same JSONL schema
  `{id, question, answer_qids: [...], bridge_qids: [...], source, cypher?, difficulty}`:
  1. **KG-pattern**: sample 2–4-hop Cypher patterns with filters, execute for exact answer sets,
     LLM-verbalize, round-trip check (LLM judges question↔pattern faithfulness), drop answer
     sets that are empty or >30.
  2. **Similarity-link** (SID-style): chain entities via abstract-embedding similarity so ≥1 hop
     is text-only (not a triple), forcing hybrid vector+graph trajectories.
- **Eval**: FRAMES (google/frames-benchmark) mapped to golden entity sets via its linked wiki
  articles → QIDs; plus held-out slices of both synthetic pipelines. Baseline harness runs a
  frontier model with the same tool server.
- **Training**: verl, GRPO (no SFT to start; optional warm-start dir exists), multi-turn tool
  calling against the tool server. Length scheduling; format reward. Policy models:
  Qwen3-0.6B (validation), Qwen3-4B (main run).

## Directory ownership (parallel workstreams — do not edit outside your dirs)

- `ingest/`, `toolserver/`, `docker-compose.yml` — workstream A (tooling + ingestion)
- `questiongen/`, `eval/` — workstream B (data + eval)
- `training/` — workstream C (RL environment)
- `src/ecs/` — shared: config, answer parser, embedder wrapper. Owned by A; B/C import only.
- `data/` — gitignored artifacts (downloads, indexes, generated questions).

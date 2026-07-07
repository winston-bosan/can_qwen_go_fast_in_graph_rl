# ingest

Pipeline order (all idempotent / resumable; run from the repo root with the venv):

```bash
docker compose up -d
.venv/bin/python ingest/download.py        # Wikidata5M -> data/wikidata5m/
.venv/bin/python ingest/build_sidecar.py   # -> data/sidecar.db
.venv/bin/python ingest/load_neo4j.py      # CSVs + neo4j-admin bulk import + index
.venv/bin/python ingest/embed_qdrant.py --limit 20000   # smoke test
.venv/bin/python ingest/embed_qdrant.py                 # full corpus (long; resumable)
```

## download.py

Fetches the transductive split from the `intfloat/wikidata5m` Hugging Face
mirror (override with `ECS_WIKIDATA5M_REPO`) and extracts:
`wikidata5m_transductive_{train,valid,test}.txt`, `wikidata5m_text.txt`,
`wikidata5m_entity.txt`, `wikidata5m_relation.txt`. Verifies line counts
(~20.6M train triples, ~4.82M text entries, 825 relations).
`--verify-only` re-checks without downloading.

## build_sidecar.py

Builds `data/sidecar.db` (sqlite):

- `entity(qid PRIMARY KEY, title, abstract)` — title = first alias; entities
  that appear only in the text file get `title = qid`.
- `alias(qid, alias)` — indexed on `alias` (name lookup) and `qid`.
- `relation(pid PRIMARY KEY, label)` — label = first alias.

Builds into `sidecar.db.part`, atomically renamed on success.

## load_neo4j.py — bulk import procedure

`neo4j-admin database import full` requires the target database to be
**offline and empty**. The script automates exactly this:

1. Generate CSVs into `data/neo4j/import/` (bind-mounted at `/import`):
   - `entities.csv` — header `qid:ID,title`, one row per entity
     (sidecar entities ∪ any qid appearing in train triples).
   - `triples.csv` — header `:START_ID,:TYPE,:END_ID,label`; `:TYPE` is the
     P-id, `label` the English relation name (denormalized onto every edge).
2. `docker compose stop neo4j` — the server must not hold the store files.
3. One-off container run (root) against the same volumes:

   ```bash
   docker compose run --rm --no-deps --entrypoint bash neo4j -c \
     "neo4j-admin database import full neo4j \
        --nodes=Entity=/import/entities.csv \
        --relationships=/import/triples.csv \
        --overwrite-destination --verbose \
      && chown -R neo4j:neo4j /data"
   ```

   `--overwrite-destination` deletes any existing `neo4j` database first, so
   re-runs don't need a fresh volume. The import tool runs as root in the
   one-off container, so the trailing `chown` hands the store back to the
   `neo4j` user the server runs as.
4. `docker compose start neo4j`, wait for bolt, then
   `CREATE INDEX entity_qid IF NOT EXISTS FOR (n:Entity) ON (n.qid)` and
   verify counts + spot-check the neighbors of Q13371 (Harvard University).

Flags: `--csv-only` (just regenerate CSVs), `--verify` (counts/spot query only).

## embed_qdrant.py

Creates the `wiki_entities` collection (640-dim for the canonical
harrier-oss-v1-270m, cosine, on-disk vectors) and upserts
`title + ". " + abstract` embeddings (abstract truncated to ~400 tokens) with
payload `{qid, title}`; point id = integer part of the QID. Progress is
checkpointed to `data/qdrant_embed_progress.json` after every batch — kill and
re-run to resume. `--limit N` for smoke tests, `--restart` to wipe and start
over. Prints measured entities/s and tokens/s. The dim guard refuses to write
if the loaded model's dim doesn't match `config.EMBED_DIM` — the checkpoint
and collection are model-specific, so switching models (e.g. to the 0.6b
upgrade path) requires `--restart` after updating the config.

Dtype note: the 270m is Gemma3-based and produces all-NaN embeddings in fp16;
`ecs.embedder` therefore uses bfloat16 on CUDA (see `ECS_EMBED_DTYPE`) and
hard-fails on non-finite vectors rather than poisoning the index.

Measured on the RTX 3060: 270m bf16 ≈ 260 ent/s (~32k tok/s) → full 4.94M
corpus in ~5.5 h. Run it detached (or on a rented GPU pointing
`ECS_QDRANT_URL` at this box). Historical, still-valid measurements for the
0.6b upgrade path (fp16, 1024-dim): 68.6 ent/s solo / ~59 ent/s with the
toolserver co-resident (~8.6k tok/s) → ~20-23 h full corpus.

## bench_llamacpp.py — llama.cpp embedding parity + throughput

Historical (2026-07, from when harrier-oss-v1-**0.6b** was the canonical
embedder; conclusions about the llama.cpp-vs-PyTorch gap should transfer to
the 270m but were not re-measured). Benchmarks a GGUF served by llama.cpp
against the sentence-transformers stack. Serve first (official CUDA image):

```bash
docker run -d --name ecs-llamacpp --gpus all -p 7802:8080 \
  -v $PWD/data/gguf:/models ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m /models/harrier-oss-v1-0.6b.Q8_0.gguf \
  --embeddings --pooling last --embd-normalize 2 \
  -ngl 99 -c 8192 -b 4096 -ub 2048 --parallel 4 --host 0.0.0.0 --port 8080
```

Then:

```bash
.venv/bin/python ingest/bench_llamacpp.py parity      # doc/query cosine ST vs llama.cpp
.venv/bin/python ingest/bench_llamacpp.py retrieval   # top-10 overlap on temp collections
.venv/bin/python ingest/bench_llamacpp.py bench       # ent/s + tokens/s
```

Tokenizer parity note: llama.cpp `add_special=true` appends the same trailing
`<|endoftext|>` (151643) as the HF tokenizer, no BOS — matching ST's
last-token pooling input exactly, so no prompt-template workarounds needed.

2026-07 result on the RTX 3060 (toolserver idle, ST run paused): parity PASSES
(mean doc cosine 0.9997, min 0.980; query 0.9998; top-10 overlap 9.7/10) but
throughput LOSES: best llama.cpp q8_0 = 35-39 ent/s (~4.5-5.0k tok/s) across
parallel/batch settings (4-32 slots, ub 1024-4096, FA on/off, f16 GGUF no
better) vs sentence-transformers fp16 = 68.6 ent/s (8.6k tok/s). PyTorch's
dense fp16 batching wins prompt-only embedding on this GPU; ST remains the
production stack.

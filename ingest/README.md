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

Creates the `wiki_entities` collection (1024-dim, cosine, on-disk vectors) and
upserts `title + ". " + abstract` embeddings (abstract truncated to ~400
tokens) with payload `{qid, title}`; point id = integer part of the QID.
Progress is checkpointed to `data/qdrant_embed_progress.json` after every
batch — kill and re-run to resume. `--limit N` for smoke tests, `--restart`
to wipe and start over. Prints measured entities/s and tokens/s.

The full 5M-entity run takes on the order of a day on an RTX 3060 — run it
detached (or on a rented GPU pointing `ECS_QDRANT_URL` at this box).

# toolserver

FastAPI server on `:7801` exposing the four retrieval tools from DESIGN.md.
Function-calling schemas (OpenAI + Anthropic style) live in `toolserver/schema.py`.

Start (after `docker compose up -d` and ingestion):

```bash
.venv/bin/uvicorn toolserver.app:app --host 0.0.0.0 --port 7801
# or: scripts/up.sh
```

The embedding model is loaded lazily on the first `/vector_search` call
(set `ECS_EMBED_MODEL=microsoft/harrier-oss-v1-270m` for the small fallback —
note the index must have been built with the same model).

## Endpoints

### GET /health

```bash
curl -s localhost:7801/health
# {"sidecar":true,"neo4j":true,"qdrant_points":20000,"qdrant":true,"ok":true}
```

### POST /vector_search — semantic entity search

```bash
curl -s localhost:7801/vector_search -H 'content-type: application/json' \
  -d '{"query": "Ivy League university in Cambridge, Massachusetts", "k": 5}'
# [{"qid":"Q13371","title":"harvard university","score":0.83,"snippet":"Harvard University is a private Ivy League research university in Cambridge, Massachusetts..."}, ...]
```

`k` <= 50. `snippet` = first ~200 chars of the abstract (from the sqlite sidecar).

### POST /get_entity — full entity record

```bash
curl -s localhost:7801/get_entity -H 'content-type: application/json' \
  -d '{"qid": "Q13371"}'
# {"qid":"Q13371","title":"harvard university","abstract":"...","aliases":[...],
#  "degree_in":8467,"degree_out":33}
```

404 for unknown QIDs, 422 for malformed ones.

### POST /get_neighbors — paginated edge listing (hub-safe)

```bash
# people educated at Harvard (incoming P69 edges), second page of 10
curl -s localhost:7801/get_neighbors -H 'content-type: application/json' \
  -d '{"qid": "Q13371", "relation": "P69", "direction": "in", "limit": 10, "offset": 10}'
# {"total":5708,"edges":[{"src":"Q193025","rel":"P69","rel_label":"educated at",
#                         "dst":"Q13371","dst_title":"harvard university"}, ...]}
```

`direction`: `out` | `in` | `both` (default `both`); `relation`: optional P-id
filter; `limit` <= 100. `total` counts all matching edges, so callers can page
through hubs safely.

### POST /find_paths — connecting paths between two entities

```bash
curl -s localhost:7801/find_paths -H 'content-type: application/json' \
  -d '{"src_qid": "Q13371", "dst_qid": "Q9095", "max_hops": 3, "limit": 5}'
# {"paths":[{"length":2,
#            "nodes":[{"qid":"Q13371","title":"harvard university"}, ...],
#            "edges":[{"src":"...","rel":"P69","rel_label":"educated at","dst":"..."}]}]}
```

Semantics: undirected `allShortestPaths` bounded by `max_hops` (<= 4),
`limit` <= 20 paths, 30s query timeout. Shortest-path semantics keep the query
hub-safe; paths longer than the shortest connection are not enumerated.

## Tests

Integration tests run against the live server + services and **skip with a
message** when anything is down:

```bash
.venv/bin/pytest toolserver/tests -v
```

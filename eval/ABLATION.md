# Tool-surface ablation: which part of the Cypher feature space matters?

Research question: what does the ideal IR / query language for entity retrieval look like?
Instead of designing an IR a priori, ablate the tool surface along a monotone expressivity
lattice (NOT the full powerset) and see where performance jumps.

Fixed: model (untrained Qwen/Qwen3-4B; optionally Qwen3.5-9B), eval set (pilot_kg.jsonl,
n=50, held out), full 4.94M index, k=50, max 15 tool calls, identical prompts modulo the
tool schema block. Measured per arm: two-tier NDCG, recall@50, F1, tool calls/episode,
tokens/episode, wall time. Pre-RL only (5h budget); trained ablations are future work.

## Arms (expressivity lattice)

| arm | tool surface | isolates |
|---|---|---|
| A0 vector-only        | vector_search                                             | resolution-only floor |
| A1 ours-4tool         | vector_search, get_entity, get_neighbors, find_paths      | current API (baseline) |
| A2 ours+intersect     | A1 + intersect_neighbors(qid_a,rel_a,qid_b,rel_b)         | value of one composed set-op |
| A3 cypher-basic       | vector_search + run_cypher[MATCH+WHERE+RETURN+LIMIT only] | declarative patterns, no pipelining |
| A4 cypher-varlen      | A3 + variable-length paths [*1..4]                        | path expressivity |
| A5 cypher-full        | vector_search + run_cypher[read-only, WITH/agg/ORDER BY]  | full query language ceiling |
| A6 cypher-noresolve   | run_cypher[full] alone, NO vector_search                  | resolution vs expressivity split |

## Reading the result

- A1 -> A5 jump location = which Cypher features matter. A3 ~= A5 => WITH/aggregation are
  IR bloat; big A5-A3 gap => pipelining is load-bearing.
- A2 - A1 = whether a single composed op captures most of the declarative win.
- A6 vs A5 = how much of "cypher is good" is actually just entity resolution.
- Efficiency (calls, tokens) counts: an IR that halves tokens at equal NDCG wins.

## Interpretation caveat (must appear on the slide)

KG-pattern questions were GENERATED from Cypher templates, so cypher arms get a
question->query translation shortcut; their advantage is an UPPER BOUND that will not
transfer to text-hop (sim_link/FRAMES) questions. The ablation still ranks features
fairly relative to each other within the lattice.

## run_cypher safety rails (all arms)

Read-only enforcement (reject CREATE|MERGE|DELETE|SET|REMOVE|DROP|CALL {} write procs),
server-side LIMIT cap (<=200 rows), transaction timeout 10s, result truncation to QIDs +
titles, per-profile clause blacklist enforced server-side (not in the prompt).

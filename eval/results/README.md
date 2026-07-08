# Results registry

`registry.jsonl` — one record per eval run, committed (unlike `data/reports/`, which is
gitignored and machine-local). Append, never rewrite history; corrections get a new record
and a note.

Target plot: **NDCG vs parameter count (log-x)**, one trend line per `condition`
(`pre-rl` = untrained-with-tools scaling line; `post-rl` = our trained checkpoints;
`frontier`/`reference`/`no-llm` = horizontal reference marks, since they have no
comparable param count or aren't in the training family).

## Comparability rules (enforce before drawing any line through points)

A trend line may only connect records that share ALL of:
- `eval_set` (canonical: `pilot_kg.jsonl`, n=50, held out from training)
- `index_coverage` = 1.0
- the same tool surface (vector_search, not the resolver stand-in)

Current registry records that VIOLATE canonical settings (kept for history, flagged in
`notes`): the 4B pre-RL number (training-distribution questions), the 9B n=1 anecdote,
both partial-index runs, and the Fable self-test (resolver stand-in).

## Standardized battery (run before plotting anything final)

For each model in {0.6B, 1.7B(+post-RL ckpt), 4B, 9B, frontier-per-key-budget}:
`training/baseline_eval.py` over the full 50-question `pilot_kg.jsonl`, full index,
identical tool schemas and k=50. Append records with `eval_set: "pilot_kg.jsonl"`,
`n: 50`, `index_coverage: 1.0`.

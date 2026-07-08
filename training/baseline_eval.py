"""Un-finetuned policy baseline: run a local HF model as the agentic policy over
a question set and report tool-calling behavior, eval metrics, and failure modes.

This answers "how does the *base* model (no RL, no SFT) behave as our policy?"
before spending GPU-hours on GRPO. It reuses the exact machinery the verl run
uses -- same tool schemas, same hermes <tool_call> format, same answer parser
and reward as training.rollout_smoke -- but sweeps a whole JSONL of questions
and aggregates.

    # behavior smoke, no data / no tool server needed (canned tool responses):
    python -m training.baseline_eval --model Qwen/Qwen3-4B --mock-tools --limit 5

    # real eval against the live tool server (:7801) once ingest is done:
    python -m training.baseline_eval --model Qwen/Qwen3-4B \
        --questions data/questions/kg_pattern.jsonl --out data/reports/qwen3-4b_base.json

Per-episode telemetry captured:
  * assistant_turns, n_tool_calls, tool histogram (which endpoints, how often)
  * gen_tokens, wall_s
  * predicted QIDs -> ndcg_two_tier / recall@k / f1 (eval.metrics, the reward
    contract) and the full training reward breakdown (compute_reward)
  * outcome: ok | empty | unparsed  (answer quality)
  * flags:   no_tools (never called a tool) | hit_max_turns | tool_error

The aggregate report also splits mean tool-calls by outcome -- i.e. "tool calls
before failure" -- so you can see whether failures are giving-up-early or
burning-the-budget.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any

import training  # noqa: F401  (sys.path setup)
from ecs.answer import parse_entities
from eval.metrics import f1_set, ndcg_two_tier, recall_at_k
from questiongen.schema import QuestionRecord, load_records
from training.data import SYSTEM_PROMPT
from training.env.reward import compute_reward
from training.env.tools import MockToolClient, ToolServerClient, get_tool_schemas
from training.rollout_smoke import (
    SAMPLE_QUESTION,
    TurnLog,
    generate_turn,
    load_policy,
    parse_tool_calls,
    to_assistant_message,
)

DEFAULT_FIXTURE = os.path.join(os.path.dirname(__file__), "tests", "fixtures", "mock_tools.json")


async def run_episode(
    tok, model, device: str, question: dict[str, Any], client: Any, *,
    max_turns: int, max_new_tokens: int, k: int, tools: list[dict], trace: bool,
) -> dict[str, Any]:
    """One agentic rollout; returns a telemetry dict (no printing unless trace)."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question["question"]},
    ]
    log: list[TurnLog] = []
    total_gen_tokens = 0
    tool_hist: Counter = Counter()
    tool_error = False
    hit_max_turns = False
    t0 = time.time()

    for turn in range(max_turns + 1):  # +1: forced final-answer turn
        text, n_tok = generate_turn(tok, model, device, messages, tools, max_new_tokens)
        total_gen_tokens += n_tok
        calls = parse_tool_calls(text) if turn < max_turns else []
        log.append(TurnLog("assistant", text, calls, n_tok))
        messages.append(to_assistant_message(text, calls))

        if not calls:
            break

        for call in calls:
            tool_hist[call["name"]] += 1
            result = await client.call(call["name"], call["arguments"])
            if isinstance(result, str) and result.lstrip().upper().startswith("ERROR"):
                tool_error = True
            log.append(TurnLog("tool", result))
            messages.append({"role": "tool", "content": result})

        if turn == max_turns - 1:
            hit_max_turns = True
            messages.append({
                "role": "user",
                "content": "You are out of tool calls. Output your final ```entities block now.",
            })

    final_text = log[-1].text if log and log[-1].role == "assistant" else ""
    answer, bridge = list(question.get("answer_qids", [])), list(question.get("bridge_qids", []))
    predicted = parse_entities(final_text, max_entities=k)
    reward = compute_reward(final_text, answer, bridge, response_tokens=total_gen_tokens)

    if not reward.parsed:
        outcome = "unparsed"
    elif reward.n_entities == 0:
        outcome = "empty"
    else:
        outcome = "ok"

    n_tool_calls = sum(tool_hist.values())
    telem = {
        "id": question.get("id"),
        "source": question.get("source", "unknown"),
        "difficulty": question.get("difficulty"),
        "assistant_turns": sum(1 for t in log if t.role == "assistant"),
        "n_tool_calls": n_tool_calls,
        "tool_hist": dict(tool_hist),
        "gen_tokens": total_gen_tokens,
        "wall_s": round(time.time() - t0, 2),
        "n_predicted": len(predicted),
        "predicted": predicted,
        "ndcg": ndcg_two_tier(predicted, set(answer), set(bridge), k=k),
        f"recall@{k}": recall_at_k(predicted, set(answer), k=k),
        "f1": f1_set(predicted, set(answer)),
        "reward_total": reward.total,
        "outcome": outcome,
        "no_tools": n_tool_calls == 0,
        "hit_max_turns": hit_max_turns,
        "tool_error": tool_error,
    }

    if trace:
        print("\n" + "=" * 78)
        print(f"[{telem['id']}] {question['question']}")
        print(f"golden: answer={answer} bridge={bridge}")
        for i, t in enumerate(log):
            head = t.text if len(t.text) <= 300 else t.text[:300] + " ...[trunc]"
            print(f"  [{i:02d}] {t.role:9s} ({t.gen_tokens}tok) {head}")
            for c in t.tool_calls:
                print(f"       -> {c['name']}({json.dumps(c['arguments'])[:160]})")
        print(f"  => outcome={outcome} tools={n_tool_calls}{dict(tool_hist)} "
              f"ndcg={telem['ndcg']:.3f} f1={telem['f1']:.3f} reward={reward.total:.3f}")

    return telem


def aggregate(rows: list[dict], k: int) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0}

    def mean(key: str, subset: list[dict] | None = None) -> float:
        rs = subset if subset is not None else rows
        return round(sum(r[key] for r in rs) / len(rs), 4) if rs else 0.0

    outcomes = Counter(r["outcome"] for r in rows)
    tool_totals: Counter = Counter()
    for r in rows:
        tool_totals.update(r["tool_hist"])
    ok_rows = [r for r in rows if r["outcome"] == "ok"]
    failed_rows = [r for r in rows if r["outcome"] != "ok"]

    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)

    def src_agg(rs: list[dict]) -> dict:
        return {
            "n": len(rs),
            "mean_ndcg": round(sum(x["ndcg"] for x in rs) / len(rs), 4),
            f"mean_recall@{k}": round(sum(x[f"recall@{k}"] for x in rs) / len(rs), 4),
            "mean_f1": round(sum(x["f1"] for x in rs) / len(rs), 4),
        }

    return {
        "n": n,
        "mean_ndcg": mean("ndcg"),
        f"mean_recall@{k}": mean(f"recall@{k}"),
        "mean_f1": mean("f1"),
        "mean_reward": mean("reward_total"),
        "outcomes": dict(outcomes),
        "outcome_rate": {o: round(c / n, 3) for o, c in outcomes.items()},
        "tool_calls": {
            "mean_per_episode": round(sum(r["n_tool_calls"] for r in rows) / n, 2),
            "mean_when_ok": mean("n_tool_calls", ok_rows),
            "mean_when_failed": mean("n_tool_calls", failed_rows),  # calls-before-failure
            "by_tool_total": dict(tool_totals.most_common()),
            "episodes_no_tools": sum(1 for r in rows if r["no_tools"]),
            "episodes_hit_max_turns": sum(1 for r in rows if r["hit_max_turns"]),
            "episodes_tool_error": sum(1 for r in rows if r["tool_error"]),
        },
        "mean_gen_tokens": mean("gen_tokens"),
        "mean_wall_s": mean("wall_s"),
        "per_source": {s: src_agg(rs) for s, rs in sorted(by_source.items())},
    }


async def run(args: argparse.Namespace) -> int:
    tools = get_tool_schemas()
    if args.mock_tools:
        client: Any = MockToolClient(args.fixture)
    else:
        client = ToolServerClient(base_url=args.toolserver_url or None)

    if args.questions:
        if not os.path.exists(args.questions):
            print(f"SKIP: questions file not found: {args.questions}")
            return 0
        records = load_records(args.questions)
        if args.limit:
            records = records[: args.limit]
        questions = [
            {"id": r.id, "question": r.question, "answer_qids": r.answer_qids,
             "bridge_qids": r.bridge_qids, "source": r.source, "difficulty": r.difficulty}
            for r in records
        ]
    else:
        questions = [SAMPLE_QUESTION]
    if not questions:
        print("SKIP: no questions")
        return 0

    tok, model, device = load_policy(args.model, args.device)
    print(f"policy={args.model}  questions={len(questions)}  "
          f"tools={'mock' if args.mock_tools else client.base_url}", file=sys.stderr)

    rows: list[dict] = []
    try:
        for i, q in enumerate(questions):
            row = await run_episode(
                tok, model, device, q, client,
                max_turns=args.max_turns, max_new_tokens=args.max_new_tokens,
                k=args.k, tools=tools, trace=args.trace,
            )
            rows.append(row)
            if not args.trace:
                print(f"  [{i + 1}/{len(questions)}] {row['id']}: outcome={row['outcome']} "
                      f"tools={row['n_tool_calls']} ndcg={row['ndcg']:.3f} "
                      f"f1={row['f1']:.3f} ({row['wall_s']}s)", file=sys.stderr)
    finally:
        await client.close()

    report = {
        "policy": f"local:{args.model}",
        "k": args.k,
        "mock_tools": args.mock_tools,
        **aggregate(rows, args.k),
        "per_question": rows,
    }

    out = args.out or os.path.join("data", "reports",
                                   f"local_{args.model.replace('/', '_')}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    a = report
    print("\n" + "=" * 78)
    print(f"{report['policy']}  n={a['n']}")
    print(f"  NDCG={a['mean_ndcg']:.4f}  recall@{args.k}={a[f'mean_recall@{args.k}']:.4f}  "
          f"F1={a['mean_f1']:.4f}  reward={a['mean_reward']:.4f}")
    print(f"  outcomes={a['outcomes']}  ({a['outcome_rate']})")
    tc = a["tool_calls"]
    print(f"  tool calls/ep={tc['mean_per_episode']}  when_ok={tc['mean_when_ok']}  "
          f"when_failed={tc['mean_when_failed']}")
    print(f"  by tool={tc['by_tool_total']}")
    print(f"  no_tools={tc['episodes_no_tools']}  hit_max_turns={tc['episodes_hit_max_turns']}  "
          f"tool_error={tc['episodes_tool_error']}")
    print(f"  mean_gen_tokens={a['mean_gen_tokens']:.0f}  mean_wall={a['mean_wall_s']:.1f}s")
    print(f"report -> {out}")
    print("=" * 78)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--questions", default=None, help="questions JSONL (default: built-in sample)")
    p.add_argument("--mock-tools", action="store_true", help="canned tool responses; no data/server needed")
    p.add_argument("--fixture", default=DEFAULT_FIXTURE)
    p.add_argument("--toolserver-url", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--out", default=None)
    p.add_argument("--trace", action="store_true", help="print full per-episode trajectories")
    args = p.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())

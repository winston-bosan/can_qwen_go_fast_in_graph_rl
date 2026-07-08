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
import re
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


# ---------------------------------------------------------------------------
# qwen3_coder tool-call support (Qwen3.5 family emits <function=..> XML, not
# hermes <tool_call>{json}</tool_call>; see training/README.md).
# ---------------------------------------------------------------------------

_QC_WRAP_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
_QC_FUNC_RE = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.DOTALL)
_QC_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.DOTALL)
_QC_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _coerce(v: str) -> Any:
    """qwen3_coder parameters arrive as strings; int-coerce k/limit/offset/etc."""
    s = v.strip()
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    return s


def parse_tool_calls_qc(text: str) -> list[dict[str, Any]]:
    """Extract qwen3_coder calls: <function=name><parameter=k>v</parameter></function>."""
    calls = []
    for m in _QC_FUNC_RE.finditer(text):
        name = m.group(1).strip()
        args = {pm.group(1).strip(): _coerce(pm.group(2)) for pm in _QC_PARAM_RE.finditer(m.group(2))}
        calls.append({"name": name, "arguments": args})
    return calls


def to_assistant_message_qc(text: str, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    content = _QC_THINK_RE.sub("", _QC_WRAP_RE.sub("", text))
    content = _QC_FUNC_RE.sub("", content).strip()
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = [
            {"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
            for c in tool_calls
        ]
    return msg


def generate_turn_qc(tok, model, device: str, messages, tools, max_new_tokens: int):
    """Greedy turn that stops at </tool_call> (Qwen3.5 won't otherwise self-terminate)."""
    import torch

    inputs = tok.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True, enable_thinking=False,
        return_tensors="pt", return_dict=True,
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            stop_strings=["</tool_call>"], tokenizer=tok,
        )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(gen_ids, skip_special_tokens=True).strip(), int(gen_ids.shape[0])


def load_policy_any(model_name: str, device: str):
    """load_policy that falls back to the multimodal class (Qwen3_5ForConditionalGeneration)."""
    import torch
    import transformers
    from transformers import AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_name)
    last = None
    for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        try:
            cls = getattr(transformers, cls_name)
            model = cls.from_pretrained(model_name, dtype=dtype).to(device).eval()
            print(f"loaded {model_name} via {cls_name} ({type(model).__name__}) on {device}", file=sys.stderr)
            return tok, model, device
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"could not load {model_name}: {last}")


async def run_episode(
    tok, model, device: str, question: dict[str, Any], client: Any, *,
    max_turns: int, max_new_tokens: int, k: int, tools: list[dict], trace: bool,
    system_suffix: str = "", tool_format: str = "hermes",
) -> dict[str, Any]:
    """One agentic rollout; returns a telemetry dict (no printing unless trace)."""
    if tool_format == "qwen3_coder":
        gen_fn, parse_fn, asst_fn = generate_turn_qc, parse_tool_calls_qc, to_assistant_message_qc
    else:
        gen_fn, parse_fn, asst_fn = generate_turn, parse_tool_calls, to_assistant_message

    system = SYSTEM_PROMPT + ("\n\n" + system_suffix if system_suffix else "")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": question["question"]},
    ]
    log: list[TurnLog] = []
    total_gen_tokens = 0
    tool_hist: Counter = Counter()
    tool_error = False
    hit_max_turns = False
    gen_s = 0.0
    tool_s = 0.0
    tool_lat_by_name: dict[str, float] = defaultdict(float)
    t0 = time.time()

    for turn in range(max_turns + 1):  # +1: forced final-answer turn
        tg = time.time()
        text, n_tok = gen_fn(tok, model, device, messages, tools, max_new_tokens)
        gen_s += time.time() - tg
        total_gen_tokens += n_tok
        calls = parse_fn(text) if turn < max_turns else []
        log.append(TurnLog("assistant", text, calls, n_tok))
        messages.append(asst_fn(text, calls))

        if not calls:
            break

        for call in calls:
            tool_hist[call["name"]] += 1
            tt = time.time()
            result = await client.call(call["name"], call["arguments"])
            dt = time.time() - tt
            tool_s += dt
            tool_lat_by_name[call["name"]] += dt
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
        "gen_s": round(gen_s, 2),
        "tool_s": round(tool_s, 2),
        "tool_s_by_name": {k: round(v, 3) for k, v in tool_lat_by_name.items()},
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
        other = round(telem['wall_s'] - gen_s - tool_s, 2)
        print(f"  LATENCY wall={telem['wall_s']}s = gen={gen_s:.2f}s ({gen_s/telem['wall_s']*100:.0f}%) "
              f"+ tools={tool_s:.2f}s ({tool_s/telem['wall_s']*100:.0f}%) + other={other}s | "
              f"per-tool: {telem['tool_s_by_name']} | {total_gen_tokens}tok @ {total_gen_tokens/max(gen_s,1e-9):.0f} tok/s")

    return telem


async def run_episode_openai(
    oai, model: str, question: dict[str, Any], client: Any, *,
    max_tool_calls: int, max_new_tokens: int, k: int, tools: list[dict],
    system_suffix: str = "",
) -> dict[str, Any]:
    """Agentic episode against an OpenAI-compatible server (vLLM). Uses the
    server's native tool_calls (format-agnostic) so no regex parsing. Returns
    the same telemetry dict shape as run_episode."""
    system = SYSTEM_PROMPT + ("\n\n" + system_suffix if system_suffix else "")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": question["question"]},
    ]
    log: list[TurnLog] = []
    total_gen_tokens = 0
    tool_hist: Counter = Counter()
    tool_lat_by_name: dict[str, float] = defaultdict(float)
    tool_error = False
    hit_budget = False
    gen_s = tool_s = 0.0
    t0 = time.time()
    calls_made = 0

    while True:
        over = calls_made >= max_tool_calls
        tg = time.time()
        resp = await oai.chat.completions.create(
            model=model, messages=messages, tools=tools,
            tool_choice="none" if over else "auto",
            max_tokens=max_new_tokens, temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        gen_s += time.time() - tg
        msg = resp.choices[0].message
        if resp.usage:
            total_gen_tokens += resp.usage.completion_tokens
        tcs = msg.tool_calls or []
        log.append(TurnLog("assistant", msg.content or "",
                           [{"name": tc.function.name, "arguments": tc.function.arguments} for tc in tcs]))
        if not tcs:
            break  # final answer
        messages.append({
            "role": "assistant", "content": msg.content,
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                           for tc in tcs],
        })
        for tc in tcs:
            calls_made += 1
            tool_hist[tc.function.name] += 1
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tt = time.time()
            result = await client.call(tc.function.name, args)
            dt = time.time() - tt
            tool_s += dt
            tool_lat_by_name[tc.function.name] += dt
            if isinstance(result, str) and result.lstrip().upper().startswith("ERROR"):
                tool_error = True
            log.append(TurnLog("tool", result))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        if over:
            hit_budget = True
            break

    final_text = log[-1].text if log and log[-1].role == "assistant" else ""
    answer, bridge = list(question.get("answer_qids", [])), list(question.get("bridge_qids", []))
    predicted = parse_entities(final_text, max_entities=k)
    reward = compute_reward(final_text, answer, bridge, response_tokens=total_gen_tokens)
    outcome = "unparsed" if not reward.parsed else ("empty" if reward.n_entities == 0 else "ok")
    n_tool_calls = sum(tool_hist.values())
    return {
        "id": question.get("id"), "source": question.get("source", "unknown"),
        "difficulty": question.get("difficulty"),
        "assistant_turns": sum(1 for t in log if t.role == "assistant"),
        "n_tool_calls": n_tool_calls, "tool_hist": dict(tool_hist),
        "gen_tokens": total_gen_tokens, "wall_s": round(time.time() - t0, 2),
        "gen_s": round(gen_s, 2), "tool_s": round(tool_s, 2),
        "tool_s_by_name": {kk: round(vv, 3) for kk, vv in tool_lat_by_name.items()},
        "n_predicted": len(predicted), "predicted": predicted,
        "ndcg": ndcg_two_tier(predicted, set(answer), set(bridge), k=k),
        f"recall@{k}": recall_at_k(predicted, set(answer), k=k),
        "f1": f1_set(predicted, set(answer)), "reward_total": reward.total,
        "outcome": outcome, "no_tools": n_tool_calls == 0,
        "hit_max_turns": hit_budget, "tool_error": tool_error,
    }


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
    if getattr(args, "arm", "ours") != "ours":
        from eval.agent_baseline import tool_schemas as _arm_schemas
        tools = _arm_schemas(args.arm)
    else:
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

    rows: list[dict] = []
    done = 0

    def _log_row(row: dict) -> None:
        nonlocal done
        done += 1
        if not args.trace:
            print(f"  [{done}/{len(questions)}] {row['id']}: outcome={row['outcome']} "
                  f"tools={row['n_tool_calls']} ndcg={row['ndcg']:.3f} "
                  f"f1={row['f1']:.3f} ({row['wall_s']}s)", file=sys.stderr)

    try:
        if args.serve_url:
            # ---- vLLM / OpenAI backend: native tool_calls, concurrent ----
            from openai import AsyncOpenAI
            oai = AsyncOpenAI(base_url=args.serve_url.rstrip("/"), api_key="EMPTY")
            served = args.served_model or args.model
            print(f"policy={served} (vllm {args.serve_url})  questions={len(questions)}  "
                  f"concurrency={args.concurrency}  tools={'mock' if args.mock_tools else client.base_url}",
                  file=sys.stderr)
            sem = asyncio.Semaphore(args.concurrency)

            async def worker(q: dict) -> dict:
                async with sem:
                    row = await run_episode_openai(
                        oai, served, q, client, max_tool_calls=args.max_tool_calls,
                        max_new_tokens=args.max_new_tokens, k=args.k, tools=tools,
                        system_suffix=args.system_suffix)
                    _log_row(row)
                    return row

            rows = await asyncio.gather(*[worker(q) for q in questions])
            policy_name = f"vllm:{served}"
        else:
            # ---- local HF transformers backend: sequential ----
            tok, model, device = load_policy_any(args.model, args.device)
            print(f"policy={args.model}  questions={len(questions)}  format={args.tool_format}  "
                  f"tools={'mock' if args.mock_tools else client.base_url}", file=sys.stderr)
            for q in questions:
                row = await run_episode(
                    tok, model, device, q, client,
                    max_turns=args.max_turns, max_new_tokens=args.max_new_tokens,
                    k=args.k, tools=tools, trace=args.trace,
                    system_suffix=args.system_suffix, tool_format=args.tool_format,
                )
                rows.append(row)
                _log_row(row)
            policy_name = f"local:{args.model}"
    finally:
        await client.close()

    report = {
        "policy": policy_name,
        "k": args.k,
        "mock_tools": args.mock_tools,
        **aggregate(rows, args.k),
        "per_question": rows,
    }

    out = args.out or os.path.join(
        "data", "reports", policy_name.replace("/", "_").replace(":", "_") + ".json")
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
    p.add_argument("--system-suffix", default="", help="extra text appended to the system prompt")
    p.add_argument("--system-suffix-file", default=None,
                   help="read the system-prompt suffix from a file (overrides --system-suffix)")
    p.add_argument("--tool-format", default="auto", choices=["auto", "hermes", "qwen3_coder"],
                   help="tool-call syntax to parse (local HF backend only); auto -> qwen3_coder for Qwen3.5* else hermes")
    # vLLM / OpenAI backend (fast, concurrent; uses the server's native tool_calls)
    p.add_argument("--serve-url", default=None,
                   help="OpenAI-compatible base_url (e.g. http://localhost:8000/v1); enables the vLLM backend")
    p.add_argument("--served-model", default=None, help="model id as served by vLLM (default: --model)")
    p.add_argument("--concurrency", type=int, default=32, help="concurrent episodes (vLLM backend)")
    p.add_argument("--max-tool-calls", type=int, default=15, help="tool-call budget per episode (vLLM backend)")
    args = p.parse_args()
    if args.system_suffix_file:
        with open(args.system_suffix_file, encoding="utf-8") as f:
            args.system_suffix = f.read().strip()
    if args.tool_format == "auto":
        args.tool_format = "qwen3_coder" if "qwen3.5" in args.model.lower() else "hermes"
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())

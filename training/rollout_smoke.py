"""Trainer-free end-to-end smoke test of the RL environment.

Loads a small HF policy (default Qwen/Qwen3-0.6B), runs the REAL multi-turn
tool-calling loop -- same tool schemas, same hermes <tool_call> format, same
answer parsing and reward function the verl run will use -- and prints the
trajectory + reward breakdown. Validates env + reward before renting GPUs.

    # against the live tool server (:7801)
    python -m training.rollout_smoke

    # no tool server / no data needed: canned tool responses from a fixture
    python -m training.rollout_smoke --mock-tools

Exit code 0 iff the loop ran to completion and a reward was computed
(a reward of 0.0 from a badly-formatted model answer still "passes" -- the
point is to exercise the machinery, not the 0.6B's zero-shot skill).
Use --require-parse to also demand a parseable ```entities block.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import training  # noqa: F401  (sys.path setup)
from training.data import SYSTEM_PROMPT
from training.env.reward import compute_reward
from training.env.tools import MockToolClient, ToolServerClient, get_tool_schemas

logger = logging.getLogger("rollout_smoke")

DEFAULT_FIXTURE = os.path.join(os.path.dirname(__file__), "tests", "fixtures", "mock_tools.json")

# Built-in sample question, consistent with tests/fixtures/mock_tools.json.
SAMPLE_QUESTION = {
    "id": "smoke-1",
    "question": "Which college did the author of 'The Hitchhiker's Guide to the Galaxy' attend?",
    "answer_qids": ["Q691283"],  # St John's College, Cambridge
    "bridge_qids": ["Q42"],  # Douglas Adams (the bridge entity)
    "source": "smoke",
    "difficulty": "easy",
}

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


@dataclass
class TurnLog:
    role: str
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    gen_tokens: int = 0


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract hermes-format tool calls: <tool_call>{"name":..,"arguments":{..}}</tool_call>."""
    calls = []
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj:
            args = obj.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append({"name": obj["name"], "arguments": args})
    return calls


def load_policy(model_name: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    logger.info("loading %s on %s (%s)", model_name, device, dtype)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()
    return tok, model, device


def generate_turn(tok, model, device: str, messages: list[dict], tools: list[dict],
                  max_new_tokens: int) -> tuple[str, int]:
    import torch

    inputs = tok.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        enable_thinking=False,  # keep the 0.6B terse; verl runs choose their own template kwargs
        return_tensors="pt",
        return_dict=True,
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()
    return text, int(gen_ids.shape[0])


def to_assistant_message(text: str, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Assistant message in the shape Qwen3's chat template re-renders correctly."""
    content = _TOOL_CALL_RE.sub("", text).strip()
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = [
            {"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
            for c in tool_calls
        ]
    return msg


async def run_episode(args: argparse.Namespace, question: dict[str, Any]) -> int:
    tools = get_tool_schemas()
    if args.mock_tools:
        client: Any = MockToolClient(args.fixture)
        logger.info("using MOCK tools from %s", args.fixture)
    else:
        client = ToolServerClient(base_url=args.toolserver_url or None)
        logger.info("using live tool server at %s", client.base_url)

    tok, model, device = load_policy(args.model, args.device)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question["question"]},
    ]
    trace: list[TurnLog] = []
    total_gen_tokens = 0
    tool_calls_made = 0
    t0 = time.time()

    try:
        for turn in range(args.max_turns + 1):  # +1: forced final-answer turn
            text, n_tok = generate_turn(tok, model, device, messages, tools, args.max_new_tokens)
            total_gen_tokens += n_tok
            calls = parse_tool_calls(text) if turn < args.max_turns else []
            trace.append(TurnLog("assistant", text, calls, n_tok))
            messages.append(to_assistant_message(text, calls))

            if not calls:
                break  # model gave (or was forced to give) its final answer

            for call in calls:
                tool_calls_made += 1
                result = await client.call(call["name"], call["arguments"])
                trace.append(TurnLog("tool", result))
                messages.append({"role": "tool", "content": result})

            if turn == args.max_turns - 1:
                messages.append({
                    "role": "user",
                    "content": "You are out of tool calls. Output your final ```entities block now.",
                })
    finally:
        await client.close()

    final_text = trace[-1].text if trace and trace[-1].role == "assistant" else ""
    result = compute_reward(
        final_text,
        question.get("answer_qids", []),
        question.get("bridge_qids", []),
        response_tokens=total_gen_tokens,
    )

    # ---- report ------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"question : {question['question']}")
    print(f"golden   : answer={question.get('answer_qids')} bridge={question.get('bridge_qids')}")
    print("-" * 78)
    for i, t in enumerate(trace):
        head = t.text if len(t.text) <= 400 else t.text[:400] + " ...[truncated]"
        print(f"[{i:02d}] {t.role:9s} ({t.gen_tokens} tok) {head}")
        for c in t.tool_calls:
            print(f"     -> tool_call {c['name']}({json.dumps(c['arguments'])[:200]})")
    print("-" * 78)
    assistant_turns = sum(1 for t in trace if t.role == "assistant")
    print(f"turns: {assistant_turns} assistant / {tool_calls_made} tool calls | "
          f"generated tokens: {total_gen_tokens} | wall: {time.time() - t0:.1f}s")
    print(f"reward: total={result.total:.4f}  task({'ndcg' if result.parsed else '-'})="
          f"{result.task_score:.4f}  format=+{result.format_bonus if result.parsed else 0:.2f}  "
          f"dump_pen=-{result.dump_penalty:.4f} (junk={result.junk_count})  "
          f"len_pen=-{result.length_penalty:.4f}  parsed={result.parsed}  "
          f"n_entities={result.n_entities}")
    print("=" * 78)

    if args.require_parse and not result.parsed:
        print("FAIL: final answer had no parseable ```entities block (--require-parse)")
        return 1
    print("SMOKE PASS: multi-turn loop + tool adapter + answer parse + reward all executed.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--mock-tools", action="store_true",
                   help="use canned tool responses from --fixture instead of the live server")
    p.add_argument("--fixture", default=DEFAULT_FIXTURE)
    p.add_argument("--toolserver-url", default=None, help="override ECS_TOOLSERVER_URL")
    p.add_argument("--max-turns", type=int, default=8, help="max tool-calling turns")
    p.add_argument("--max-new-tokens", type=int, default=512, help="per-turn generation cap")
    p.add_argument("--question-file", default=None,
                   help="JSONL of question records; uses the first record (default: built-in sample)")
    p.add_argument("--require-parse", action="store_true",
                   help="exit nonzero unless the final answer parses to >=1 QID")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    question = SAMPLE_QUESTION
    if args.question_file:
        with open(args.question_file) as f:
            question = json.loads(next(line for line in f if line.strip()))

    return asyncio.run(run_episode(args, question))


if __name__ == "__main__":
    sys.exit(main())

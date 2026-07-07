"""Toolserver load test: verify QPS on the pod BEFORE starting a training run.

Mimics one GRPO rollout step's tool traffic: ~70% vector_search / 20%
get_neighbors / 10% get_entity (find_paths is rare in trajectories and
excluded from the default mix; add it via --mix). QIDs for the graph calls are
harvested live from vector_search responses, so traffic follows realistic
key distributions instead of hammering one hot entity.

    # 30s at 64 concurrent requests against $ECS_TOOLSERVER_URL
    python training/launch/loadtest_toolserver.py --duration 30 --concurrency 64

    # fixed request count, custom mix including find_paths
    python training/launch/loadtest_toolserver.py --requests 3000 --mix 65,20,10,5

Interpretation: a training step needs ~1.5k tool calls (validate config,
32 q x 8 rollouts x ~6 calls) to ~6k (main config, 64 x 16 x ~6). The report
projects the wall-clock those volumes would take at the achieved QPS.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time
from collections import defaultdict

import httpx

# Varied, entity-flavoured queries so vector_search work is realistic
# (embedding + HNSW) and returns diverse QIDs for the graph-call pool.
QUERIES = [
    "English author of comic science fiction novels",
    "college of the University of Cambridge",
    "Nobel laureate in physics who worked on quantum electrodynamics",
    "capital city on the Danube river",
    "American technology company founded in a garage",
    "German composer of the Baroque period",
    "mountain range between France and Spain",
    "programming language created at Bell Labs",
    "ancient Greek philosopher who taught Alexander the Great",
    "Japanese animation film studio",
    "spacecraft that first landed humans on the Moon",
    "river flowing through Egypt",
    "painter of the Sistine Chapel ceiling",
    "island nation in the North Atlantic with geysers",
    "chemical element with atomic number 79",
    "British rock band formed in Liverpool",
    "theory describing gravity as spacetime curvature",
    "largest desert in Africa",
    "founder of psychoanalysis",
    "medieval fortress in London on the Thames",
    "particle accelerator laboratory near Geneva",
    "playwright of Hamlet and Macbeth",
    "first woman to win a Nobel Prize",
    "video game company that created Mario",
    "cathedral in Paris damaged by fire in 2019",
]

FALLBACK_QIDS = ["Q42", "Q25169", "Q691283", "Q35794", "Q145", "Q36180"]


class Stats:
    def __init__(self) -> None:
        self.lat: dict[str, list[float]] = defaultdict(list)
        self.errors: dict[str, int] = defaultdict(int)

    def record(self, ep: str, dt: float, ok: bool) -> None:
        if ok:
            self.lat[ep].append(dt)
        else:
            self.errors[ep] += 1

    def report(self, wall: float) -> None:
        total_ok = sum(len(v) for v in self.lat.values())
        total_err = sum(self.errors.values())
        qps = total_ok / wall if wall > 0 else 0.0
        print(f"\n{'endpoint':<15}{'n':>7}{'err':>6}{'p50 ms':>9}{'p90 ms':>9}{'p99 ms':>9}{'max ms':>9}")
        for ep in sorted(set(self.lat) | set(self.errors)):
            v = sorted(self.lat.get(ep, []))
            if v:
                q = lambda p: 1000 * v[min(len(v) - 1, int(p * len(v)))]  # noqa: E731
                print(f"{ep:<15}{len(v):>7}{self.errors.get(ep, 0):>6}"
                      f"{q(0.50):>9.1f}{q(0.90):>9.1f}{q(0.99):>9.1f}{1000 * v[-1]:>9.1f}")
            else:
                print(f"{ep:<15}{0:>7}{self.errors.get(ep, 0):>6}")
        print(f"\ntotal: {total_ok} ok / {total_err} errors in {wall:.1f}s -> {qps:.1f} QPS")
        if qps > 0:
            print(f"projected wall per training step:")
            print(f"  validate (1.5k calls/step): {1500 / qps:6.1f}s")
            print(f"  main     (6k   calls/step): {6000 / qps:6.1f}s"
                  f"   {'OK' if 6000 / qps < 120 else 'WARNING: >2min/step, tool calls may dominate'}")
        if total_err:
            print("WARNING: errors occurred -- during training these surface as "
                  "'ERROR:' tool outputs and add reward noise.")


async def worker(
    client: httpx.AsyncClient,
    stats: Stats,
    qid_pool: list[str],
    mix: list[float],
    deadline: float | None,
    remaining: list[int],
    rng: random.Random,
) -> None:
    endpoints = ["vector_search", "get_neighbors", "get_entity", "find_paths"]
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            return
        if remaining[0] <= 0 and deadline is None:
            return
        remaining[0] -= 1

        ep = rng.choices(endpoints[: len(mix)], weights=mix, k=1)[0]
        if ep == "vector_search":
            body: dict = {"query": rng.choice(QUERIES), "k": rng.choice([5, 10, 10, 20])}
        elif ep == "get_neighbors":
            body = {"qid": rng.choice(qid_pool), "direction": "both", "limit": 25}
        elif ep == "get_entity":
            body = {"qid": rng.choice(qid_pool)}
        else:  # find_paths
            body = {"src_qid": rng.choice(qid_pool), "dst_qid": rng.choice(qid_pool), "max_hops": 3}

        t0 = time.monotonic()
        ok = False
        try:
            r = await client.post(f"/{ep}", json=body)
            ok = r.status_code == 200
            if ep == "vector_search" and ok:
                for hit in r.json():
                    q = hit.get("qid")
                    if q and len(qid_pool) < 5000:
                        qid_pool.append(q)
        except httpx.HTTPError:
            ok = False
        stats.record(ep, time.monotonic() - t0, ok)


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=os.environ.get("ECS_TOOLSERVER_URL", "http://localhost:7801"))
    p.add_argument("--concurrency", type=int, default=64,
                   help="parallel in-flight requests (rollout does ~batch*group trajectories concurrently)")
    p.add_argument("--duration", type=float, default=None, help="seconds to run (default: --requests mode)")
    p.add_argument("--requests", type=int, default=2000, help="total requests when --duration not given")
    p.add_argument("--mix", default="70,20,10",
                   help="percent weights: vector_search,get_neighbors,get_entity[,find_paths]")
    p.add_argument("--warmup", type=int, default=8,
                   help="sequential vector_search warmups (loads the lazy embedder in each uvicorn worker)")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    mix = [float(x) for x in args.mix.split(",")]
    if not 3 <= len(mix) <= 4:
        p.error("--mix needs 3 or 4 comma-separated weights")
    rng = random.Random(args.seed)
    stats = Stats()
    qid_pool: list[str] = list(FALLBACK_QIDS)

    async with httpx.AsyncClient(
        base_url=args.url,
        timeout=args.timeout,
        limits=httpx.Limits(max_connections=args.concurrency + 8),
    ) as client:
        try:
            r = await client.get("/health")
            print(f"health: {r.json()}")
        except httpx.HTTPError as e:
            print(f"FATAL: toolserver unreachable at {args.url}: {e}")
            return 1

        print(f"warmup: {args.warmup} vector_search calls (embedder load can take ~30s/worker)")
        for i in range(args.warmup):
            try:
                r = await client.post("/vector_search", json={"query": QUERIES[i % len(QUERIES)], "k": 5})
                if r.status_code == 200:
                    qid_pool.extend(h["qid"] for h in r.json() if h.get("qid"))
            except httpx.HTTPError as e:
                print(f"warmup call failed: {e}")

        print(f"load: concurrency={args.concurrency} mix={mix} "
              f"{'duration=' + str(args.duration) + 's' if args.duration else 'requests=' + str(args.requests)}")
        deadline = time.monotonic() + args.duration if args.duration else None
        remaining = [args.requests if not args.duration else 1 << 62]
        t0 = time.monotonic()
        await asyncio.gather(*(
            worker(client, stats, qid_pool, mix, deadline, remaining, random.Random(args.seed + i))
            for i in range(args.concurrency)
        ))
        stats.report(time.monotonic() - t0)

    total_err = sum(stats.errors.values())
    total_ok = sum(len(v) for v in stats.lat.values())
    return 0 if total_ok > 0 and total_err / max(1, total_ok + total_err) < 0.01 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

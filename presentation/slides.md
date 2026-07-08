---
title: "NLP in, IR out"
sub_title: "or: can Qwen go fast in graph RL?"
author: entity_component_search — 2 days, 1 A100, 1 RTX 3060
---

The ask
===

> **"Natural language in, intermediate representation out."**

Given a question and a knowledge base, emit *something* — a query, a plan, a
program — that retrieves the entities needed to answer it.

```
"Find all Harvard grads that ended up going to YC and selling in 2026"
                              |
                              v
                        ¯\_(ツ)_/¯                     <- the IR
                              |
                              v
        [Q123 Paul Graham, Q456 Sam Altman, ...]
```

<!-- pause -->

The catch: **nobody knows what the IR should look like.**
Cypher? SQL? A tool-call DAG? Some learned latent thing?

<!-- end_slide -->

Operationalizing the ask
===

Concrete subproblems we carved out of the vibe:

1. **What is the best feature set of an IR?**
   Which parts of a query language actually earn their keep?
2. **Can we stay under ~1s for generation + query?**
   An IR is useless for search if it's slow to emit or slow to run.
3. **How do we *score* an IR at all?**
   Need a reward: exact golden entity sets, per question, at scale.
4. **Where does training data come from?**
   No human labels. No existing dataset with our shape.

<!-- pause -->

**And the strategic decision that made the 2 days survivable:**

> Don't design the IR. Train a small model end-to-end with RL against
> retrieval reward — then **read the IR off its trajectories.**

(Recipe: SID-1 technical report — GRPO, no SFT, NDCG reward. We swap
"documents" for "entities" and give it a graph.)

<!-- end_slide -->

The rig
===

**Corpus**: Wikidata5M — 4.94M entities, 20.6M triples, aligned Wikipedia abstracts

**Databases**: neo4j (graph) + qdrant (640-dim vectors, harrier-270m embedder)

**Tool surface** (deliberately tiny — 4 tools, no Cypher for the model):

| tool | does | p50 latency |
|---|---|---|
| `vector_search(q, k)` | name/description -> entities | ~50ms warm |
| `get_entity(qid)` | title, abstract, degrees | ~10ms |
| `get_neighbors(qid, rel?, dir)` | typed edges, paginated | ~20ms |
| `find_paths(a, b, max_hops)` | bounded shortest paths | ~100ms |

All bounded, all paginated, hub-safe. Latency budget: **~1s/call hard ceiling.**

<!-- pause -->

Why no raw Cypher for the policy? Three reasons:
- **cache**: fixed parameterized queries keep neo4j's page cache warm
- **reward hacking**: golden sets are *generated* from Cypher — a Cypher tool
  turns the task into question->query translation
- **the trajectories are the research artifact** — primitives force visible planning

<!-- end_slide -->

Training data: golden sets for free
===

The KG inverts the labeling problem: **sample a Cypher pattern, execute it,
then have an LLM verbalize it** — every question ships with an exact answer set.

```
MATCH (p)-[:P108]->(c)-[:P112]->(f)-[:P69]->(school {qid:'Q371625'})
  -> executes to 7 people
  -> "Name the people who work for a company owned by a parent
      organization founded by someone who attended Brooklyn College."
      (answer: 7 DreamWorks Animation staff, via David Geffen)
```

**7,019 questions**, deepseek-v4-pro via OpenRouter, **$0.0028 / accepted question**

<!-- pause -->

Quality control that actually caught things:

- round-trip judge: blind model must reconstruct the relation multiset (49/49)
- **Claude-panel calibration on 50 questions: 88% accept**; every reject was
  entity-level — "educated at *Yale Banner*" (a yearbook!), "graduate of" vs P69
- fixes: anchor-type gate (abstract snippet + reject sentinel),
  title hygiene filter, verbalizer constraints

<!-- end_slide -->

The reward (and its measured blind spot)
===

```
reward = max(0,  NDCG_two_tier@50  +  0.1*format  -  0.2*(junk/50))
         gain 2 = answer entity, gain 1 = bridge entity
         hard 0 if the ```entities block doesn't parse
```

Why this shape:

- **coverage** dominates (recall lives inside NDCG's numerator)
- **ranking** pressure only where it's meaningful: answers > bridges > junk
  (within the answer set, order is free — no fake ranking pressure on sets)
- **speed** is a *constraint* (turn cap + length schedule), not a reward term —
  RL will happily trade recall for latency if you price it in too early

<!-- pause -->

**Measured before training** (not assumed): two-tier NDCG gives trailing junk
a penalty of exactly **0.0** — appending 40 garbage entities after 5 correct
ones costs nothing. Hence the explicit dump penalty. Reward functions have
bugs like code does; we unit-tested ours.

<!-- end_slide -->

Baselines (all on the full 4.94M index)
===

| policy (untrained) | n | NDCG | recall@50 |
|---|---|---|---|
| vector search only | 50 | 0.073 | 0.053 |
| Qwen3-0.6B + tools | 50 | 0.060 | **0.000** |
| Qwen3-4B + tools | 200 | 0.188 | 0.134 |
| Qwen3.5-9B + tools | 1 | (0.86) | (1.0) — anecdote |
| Claude (Fable) + tools | 8 | 0.976 | 1.000 |

<!-- pause -->

Two findings worth the slide space:

- The 0.6B's recall is **literally zero** — but not from coverage: it emits
  *placeholder QIDs* (`Q123`) instead of copying real ones from tool output.
  vector_search handed it the right answer; it answered `Q123` anyway.
  **"Format yes, grounding no" — that gap is exactly what RL gets to close.**
- Failed 4B episodes burn 15 tool calls; successful ones use 5.
  Untrained models *flail*, they don't abstain.

<!-- end_slide -->

The run: 60 steps of GRPO, Qwen3-1.7B, one A100
===

verl 0.8 + sglang, multi-turn tool rollouts (batch 32 x group 8 = 256/step),
length schedule 2048 -> 4096, ~170s/step.

```
train reward   0.07 ─────► 0.23   (steps 1-31, still climbing)
response len   1,900 ─────► 1,300 tokens   (less waffle, more reward)
tool turns     ~5.5 ─────► ~4.2            (fewer, better-aimed calls)
```

<!-- pause -->

**Held-out validation** (greedy decoding, questions never trained on):

```
step 0:    0.129          <- untrained 1.7B
step ~15:  0.180
step 30:   0.218          <- +69%, still rising into stage 2
```

<!-- pause -->

**The headline: after ~90 minutes of RL, the 1.7B passes the untrained 4B
(0.188) — a model 2.4x its size.** Total training cost: about the price of lunch.

<!-- end_slide -->

What we cut, and what cut us
===

**Cut for scope (deliberately):**
- the IR-design question itself — sidestepped via RL (that was the plan)
- full 15k corpus (7k is plenty for a validation run)
- trained ablations per tool-variant (pre-RL ablations only)

<!-- pause -->

**Failed honestly (documented, reproducible):**

- **sim_link v1 questions came out degenerate** — the verbalizer leaked the
  retrieval mechanism: *"Which entity appears in the Wikipedia summary of X,
  is closely associated, yet shares no direct link..."* Nobody asks that.
  Needs reformulation, not scale. (30 questions burned, not 3,000.)
- **Qwen3.5 cannot be RL-trained today**: its gated-delta-net layers hit a
  CUDA illegal-memory-access under FSDP in *both* the naive torch kernel and
  the fused fla kernel (upstream verl#6549). Inference fine; training dead.
  We burned a stack upgrade finding this. 2B/0.8B shelved -> Qwen3-1.7B.
- **The Cypher-feature ablation lattice is circular on our data**: our
  questions were *generated from* 3 Cypher clause shapes, so the ablation
  would mostly measure what our generator needs. Harder questions first.

<!-- end_slide -->

Threats to validity (read before believing anything)
===

1. **Heavily fitted to one multihop variant.** Star/chain/intersection
   patterns over 10 relations. Real questions are messier and often not
   graph-shaped at all.
2. **neo4j is not the prod graph DB.** A different engine = different query
   planner = different latency profile = possibly different optimal tool
   surface. Our ~1s budget holds *here*; that's all we can claim.
3. **The workload isn't real search traffic.** QIDs-in-context is gnarly:
   for large membership filtering ("all 445 alumni x all their employers")
   the pool outgrows the context — a *working memory* / set-handle
   abstraction may be unavoidable at scale. We punted on it, visibly.
4. **Query profile mismatch**: our questions have exact, small answer sets.
   Real queries are vague, underspecified, and have fuzzy relevance.
5. **Unknown unknowns.** The eval set and the training set share a generator.
   Every number above should be re-earned on FRAMES-style external questions.

<!-- end_slide -->

What's next
===

- **Two-arm diff** (same model, same questions): our 4 primitives vs
  vector_search + one general read-only Cypher tool — endpoint is built,
  125ms/query, waiting on eval-model compute. The honest first cut at
  "which features matter."
- **Harder questions**: sim-link v2 — chains through *embedding-similarity*
  edges (a la SID / FRAMES-without-hyperlinks), so questions stop being
  answerable by graph traversal alone.
- **4B main run**: data bundle is tarred (14GB), repo is on GitHub, stack is
  qualified — needs a bigger box (8xH100 or B300) and ~a day.
- **Trained ablations** on the winning tool surfaces (the real IR answer).
- **Working memory** for set operations — the membership-filtering problem
  above is a design question, not a training question.

<!-- end_slide -->

<!-- jump_to_middle -->

fin
===

**repo**: github.com/winston-bosan/can_qwen_go_fast_in_graph_rl

*One question, two days, three model families, seven infrastructure bugs,
one climbing reward curve.*

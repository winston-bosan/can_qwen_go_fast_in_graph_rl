"""KG-pattern question pipeline over the Wikidata5M graph in neo4j.

A library of parameterized Cypher pattern templates (2-4 hops: chains, stars,
intersections) over the schema `(:Entity {qid,title})-[:P69|P108|...]->()`.

Pipeline: `sample_pattern()` instantiates a template by sampling anchor
entities from neo4j; `execute()` runs the pattern for the exact answer set
(bridge = intermediate + anchor entities); results with empty or >30 answer
sets are dropped.

Service guarding: every neo4j call goes through `run_cypher`, which raises
`Neo4jUnavailable` (with a human-readable skip message) when the graph is not
reachable. `sample_pattern` / `execute` accept an injectable `run_query`
callable so all logic is unit-testable against in-memory fixtures.

Relation vocabulary used (Wikidata P-ids):
    P69  educated at          P108 employer          P112 founded by
    P17  country              P106 occupation        P50  author
    P57  director             P161 cast member       P463 member of
    P127 owned by
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ecs import config

from .schema import QuestionRecord

MAX_ANSWERS = 30
# Bound on the sampler's anchor candidate pool. Large enough that (template,
# anchors) dedupe over a 15k-question run doesn't exhaust any template's pool
# (single-anchor templates can emit at most this many distinct questions).
ANCHOR_POOL_LIMIT = 20000

REL_LABELS: dict[str, str] = {
    "P69": "educated at",
    "P108": "employer",
    "P112": "founded by",
    "P17": "country",
    "P106": "occupation",
    "P50": "author",
    "P57": "director",
    "P161": "cast member",
    "P463": "member of",
    "P127": "owned by",
}

RunQuery = Callable[[str, dict], list[dict]]


class Neo4jUnavailable(RuntimeError):
    """Raised when neo4j cannot be reached; message is the skip explanation."""


@dataclass(frozen=True)
class PatternTemplate:
    name: str
    kind: str  # "chain" | "star" | "intersection"
    hops: int  # number of graph edges == difficulty
    relations: tuple[str, ...]  # P-ids traversed (with multiplicity)
    anchors: tuple[str, ...]  # anchor slot names, e.g. ("school",)
    sampler_cypher: str  # one row: qid_<anchor>, title_<anchor> per anchor
    exec_cypher: str  # params $qid_<anchor>; RETURN answers, bridges
    semantics: str  # NL scaffold; "{<anchor>}" placeholders take titles


@dataclass(frozen=True)
class InstantiatedPattern:
    template: PatternTemplate
    anchors: dict[str, dict[str, str]]  # slot -> {"qid": ..., "title": ...}

    @property
    def params(self) -> dict[str, str]:
        return {f"qid_{slot}": a["qid"] for slot, a in self.anchors.items()}

    @property
    def anchor_qids(self) -> list[str]:
        return [a["qid"] for a in self.anchors.values()]

    @property
    def cypher(self) -> str:
        """Execution cypher with anchor params substituted (for the record)."""
        return render_cypher(self.template.exec_cypher, self.params)

    @property
    def semantics(self) -> str:
        """NL scaffold with anchor titles substituted."""
        return fill_semantics(self.template, self.anchors)

    @property
    def id(self) -> str:
        key = self.template.name + "|" + "|".join(sorted(self.anchor_qids))
        return "kg-" + hashlib.sha1(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Template library
# ---------------------------------------------------------------------------


def _t(**kw) -> PatternTemplate:
    return PatternTemplate(**kw)


PATTERNS: list[PatternTemplate] = [
    # ------------------------------------------------------------- 2 hops
    _t(
        name="chain_film_director_school",
        kind="chain",
        hops=2,
        relations=("P57", "P69"),
        anchors=("school",),
        sampler_cypher=f"""
MATCH (film:Entity)-[:P57]->(d:Entity)-[:P69]->(school:Entity)
WHERE $seed IS NULL OR school.qid = $seed
WITH DISTINCT school LIMIT {ANCHOR_POOL_LIMIT}
RETURN school.qid AS qid_school, school.title AS title_school
ORDER BY rand() LIMIT 1
""",
        exec_cypher="""
MATCH (film:Entity)-[:P57]->(d:Entity)-[:P69]->(school:Entity {qid: $qid_school})
RETURN collect(DISTINCT film.qid) AS answers, collect(DISTINCT d.qid) AS bridges
""",
        semantics=(
            "Find the films (answer entities) whose director (bridge) was "
            "educated at {school}."
        ),
    ),
    _t(
        name="chain_work_author_org",
        kind="chain",
        hops=2,
        relations=("P50", "P463"),
        anchors=("org",),
        sampler_cypher=f"""
MATCH (w:Entity)-[:P50]->(a:Entity)-[:P463]->(org:Entity)
WHERE $seed IS NULL OR org.qid = $seed
WITH DISTINCT org LIMIT {ANCHOR_POOL_LIMIT}
RETURN org.qid AS qid_org, org.title AS title_org
ORDER BY rand() LIMIT 1
""",
        exec_cypher="""
MATCH (w:Entity)-[:P50]->(a:Entity)-[:P463]->(org:Entity {qid: $qid_org})
RETURN collect(DISTINCT w.qid) AS answers, collect(DISTINCT a.qid) AS bridges
""",
        semantics=(
            "Find the written works (answer entities) whose author (bridge) "
            "is a member of {org}."
        ),
    ),
    _t(
        name="chain_person_company_founder",
        kind="chain",
        hops=2,
        relations=("P108", "P112"),
        anchors=("founder",),
        sampler_cypher=f"""
MATCH (p:Entity)-[:P108]->(c:Entity)-[:P112]->(founder:Entity)
WHERE $seed IS NULL OR founder.qid = $seed
WITH DISTINCT founder LIMIT {ANCHOR_POOL_LIMIT}
RETURN founder.qid AS qid_founder, founder.title AS title_founder
ORDER BY rand() LIMIT 1
""",
        exec_cypher="""
MATCH (p:Entity)-[:P108]->(c:Entity)-[:P112]->(founder:Entity {qid: $qid_founder})
RETURN collect(DISTINCT p.qid) AS answers, collect(DISTINCT c.qid) AS bridges
""",
        semantics=(
            "Find the people (answer entities) employed by an organization "
            "(bridge) that was founded by {founder}."
        ),
    ),
    _t(
        name="star_school_occupation",
        kind="star",
        hops=2,
        relations=("P69", "P106"),
        anchors=("school", "occ"),
        sampler_cypher=f"""
MATCH (occ:Entity)<-[:P106]-(p:Entity)-[:P69]->(school:Entity)
WHERE $seed IS NULL OR school.qid = $seed
WITH DISTINCT school, occ LIMIT {ANCHOR_POOL_LIMIT}
RETURN school.qid AS qid_school, school.title AS title_school,
       occ.qid AS qid_occ, occ.title AS title_occ
ORDER BY rand() LIMIT 1
""",
        exec_cypher="""
MATCH (occ:Entity {qid: $qid_occ})<-[:P106]-(p:Entity)-[:P69]->(school:Entity {qid: $qid_school})
RETURN collect(DISTINCT p.qid) AS answers, [] AS bridges
""",
        semantics=(
            "Find the people (answer entities) who were educated at {school} "
            "and whose occupation is {occ}."
        ),
    ),
    _t(
        name="star_costars",
        kind="star",
        hops=2,
        relations=("P161", "P161"),
        anchors=("film_a", "film_b"),
        sampler_cypher=f"""
MATCH (film_a:Entity)-[:P161]->(x:Entity)<-[:P161]-(film_b:Entity)
WHERE film_a.qid < film_b.qid AND ($seed IS NULL OR film_a.qid = $seed)
WITH DISTINCT film_a, film_b LIMIT {ANCHOR_POOL_LIMIT}
RETURN film_a.qid AS qid_film_a, film_a.title AS title_film_a,
       film_b.qid AS qid_film_b, film_b.title AS title_film_b
ORDER BY rand() LIMIT 1
""",
        exec_cypher="""
MATCH (fa:Entity {qid: $qid_film_a})-[:P161]->(x:Entity)<-[:P161]-(fb:Entity {qid: $qid_film_b})
RETURN collect(DISTINCT x.qid) AS answers, [] AS bridges
""",
        semantics=(
            "Find the performers (answer entities) who are cast members of "
            "both {film_a} and {film_b}."
        ),
    ),
    _t(
        name="star_director_cast",
        kind="star",
        hops=2,
        relations=("P57", "P161"),
        anchors=("director", "actor"),
        sampler_cypher=f"""
MATCH (director:Entity)<-[:P57]-(film:Entity)-[:P161]->(actor:Entity)
WHERE director.qid <> actor.qid AND ($seed IS NULL OR director.qid = $seed)
WITH DISTINCT director, actor LIMIT {ANCHOR_POOL_LIMIT}
RETURN director.qid AS qid_director, director.title AS title_director,
       actor.qid AS qid_actor, actor.title AS title_actor
ORDER BY rand() LIMIT 1
""",
        exec_cypher="""
MATCH (d:Entity {qid: $qid_director})<-[:P57]-(film:Entity)-[:P161]->(a:Entity {qid: $qid_actor})
RETURN collect(DISTINCT film.qid) AS answers, [] AS bridges
""",
        semantics=(
            "Find the films (answer entities) directed by {director} that "
            "feature {actor} as a cast member."
        ),
    ),
    # ------------------------------------------------------------- 3 hops
    _t(
        name="intersection_founder_school_owner",
        kind="intersection",
        hops=3,
        relations=("P112", "P69", "P127"),
        anchors=("school", "owner"),
        sampler_cypher=f"""
MATCH (school:Entity)<-[:P69]-(x:Entity)<-[:P112]-(y:Entity)-[:P127]->(owner:Entity)
WHERE $seed IS NULL OR school.qid = $seed
WITH DISTINCT school, owner LIMIT {ANCHOR_POOL_LIMIT}
RETURN school.qid AS qid_school, school.title AS title_school,
       owner.qid AS qid_owner, owner.title AS title_owner
ORDER BY rand() LIMIT 1
""",
        exec_cypher="""
MATCH (school:Entity {qid: $qid_school})<-[:P69]-(x:Entity)<-[:P112]-(y:Entity)-[:P127]->(owner:Entity {qid: $qid_owner})
RETURN collect(DISTINCT y.qid) AS answers, collect(DISTINCT x.qid) AS bridges
""",
        semantics=(
            "Find the organizations (answer entities) that were founded by "
            "someone (bridge) educated at {school} AND are owned by {owner}."
        ),
    ),
    _t(
        name="chain_employee_founder_school",
        kind="chain",
        hops=3,
        relations=("P108", "P112", "P69"),
        anchors=("school",),
        sampler_cypher=f"""
MATCH (p:Entity)-[:P108]->(c:Entity)-[:P112]->(f:Entity)-[:P69]->(school:Entity)
WHERE $seed IS NULL OR school.qid = $seed
WITH DISTINCT school LIMIT {ANCHOR_POOL_LIMIT}
RETURN school.qid AS qid_school, school.title AS title_school
ORDER BY rand() LIMIT 1
""",
        exec_cypher="""
MATCH (p:Entity)-[:P108]->(c:Entity)-[:P112]->(f:Entity)-[:P69]->(school:Entity {qid: $qid_school})
RETURN collect(DISTINCT p.qid) AS answers,
       collect(DISTINCT c.qid) + collect(DISTINCT f.qid) AS bridges
""",
        semantics=(
            "Find the people (answer entities) employed by an organization "
            "(bridge) whose founder (bridge) was educated at {school}."
        ),
    ),
    _t(
        name="intersection_org_member_school_country",
        kind="intersection",
        hops=3,
        relations=("P463", "P69", "P17"),
        anchors=("org", "country"),
        sampler_cypher=f"""
MATCH (org:Entity)<-[:P463]-(p:Entity)-[:P69]->(s:Entity)-[:P17]->(country:Entity)
WHERE $seed IS NULL OR org.qid = $seed
WITH DISTINCT org, country LIMIT {ANCHOR_POOL_LIMIT}
RETURN org.qid AS qid_org, org.title AS title_org,
       country.qid AS qid_country, country.title AS title_country
ORDER BY rand() LIMIT 1
""",
        exec_cypher="""
MATCH (org:Entity {qid: $qid_org})<-[:P463]-(p:Entity)-[:P69]->(s:Entity)-[:P17]->(country:Entity {qid: $qid_country})
RETURN collect(DISTINCT p.qid) AS answers, collect(DISTINCT s.qid) AS bridges
""",
        semantics=(
            "Find the people (answer entities) who are members of {org} and "
            "were educated at an institution (bridge) located in {country}."
        ),
    ),
    # ------------------------------------------------------------- 4 hops
    _t(
        name="chain_same_school_directors",
        kind="chain",
        hops=4,
        relations=("P57", "P69", "P69", "P57"),
        anchors=("film",),
        sampler_cypher=f"""
MATCH (f1:Entity)-[:P57]->(d1:Entity)-[:P69]->(s:Entity)<-[:P69]-(d2:Entity)<-[:P57]-(film:Entity)
WHERE d1 <> d2 AND f1 <> film AND ($seed IS NULL OR film.qid = $seed)
WITH DISTINCT film LIMIT {ANCHOR_POOL_LIMIT}
RETURN film.qid AS qid_film, film.title AS title_film
ORDER BY rand() LIMIT 1
""",
        exec_cypher="""
MATCH (film:Entity {qid: $qid_film})-[:P57]->(d2:Entity)-[:P69]->(s:Entity)<-[:P69]-(d1:Entity)<-[:P57]-(f1:Entity)
WHERE d1 <> d2 AND f1.qid <> $qid_film
RETURN collect(DISTINCT f1.qid) AS answers,
       collect(DISTINCT d1.qid) + collect(DISTINCT d2.qid) + collect(DISTINCT s.qid) AS bridges
""",
        semantics=(
            "Find the other films (answer entities) directed by people "
            "(bridge) who were educated at the same institution (bridge) as "
            "the director (bridge) of {film}."
        ),
    ),
    _t(
        name="chain_employee_subsidiary_founder_school",
        kind="chain",
        hops=4,
        relations=("P108", "P127", "P112", "P69"),
        anchors=("school",),
        sampler_cypher=f"""
MATCH (p:Entity)-[:P108]->(c:Entity)-[:P127]->(parent:Entity)-[:P112]->(f:Entity)-[:P69]->(school:Entity)
WHERE $seed IS NULL OR school.qid = $seed
WITH DISTINCT school LIMIT {ANCHOR_POOL_LIMIT}
RETURN school.qid AS qid_school, school.title AS title_school
ORDER BY rand() LIMIT 1
""",
        exec_cypher="""
MATCH (p:Entity)-[:P108]->(c:Entity)-[:P127]->(parent:Entity)-[:P112]->(f:Entity)-[:P69]->(school:Entity {qid: $qid_school})
RETURN collect(DISTINCT p.qid) AS answers,
       collect(DISTINCT c.qid) + collect(DISTINCT parent.qid) + collect(DISTINCT f.qid) AS bridges
""",
        semantics=(
            "Find the people (answer entities) who work for a company "
            "(bridge) owned by a parent organization (bridge) that was "
            "founded by someone (bridge) educated at {school}."
        ),
    ),
]

TEMPLATES_BY_NAME: dict[str, PatternTemplate] = {t.name: t for t in PATTERNS}


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable, no services)
# ---------------------------------------------------------------------------


def fill_semantics(
    template: PatternTemplate, anchors: dict[str, dict[str, str]]
) -> str:
    out = template.semantics
    for slot, a in anchors.items():
        out = out.replace("{" + slot + "}", f"\"{a['title']}\" ({a['qid']})")
    return out


def render_cypher(exec_cypher: str, params: dict[str, str]) -> str:
    """Substitute $params textually (for the stored record's `cypher` field)."""
    out = exec_cypher
    # longest names first so $qid_film_a is not clobbered by $qid_film
    for name in sorted(params, key=len, reverse=True):
        out = out.replace("$" + name, f"'{params[name]}'")
    return out.strip()


def anchors_from_row(template: PatternTemplate, row: dict) -> dict[str, dict[str, str]]:
    anchors: dict[str, dict[str, str]] = {}
    for slot in template.anchors:
        qid = row.get(f"qid_{slot}")
        if not qid:
            raise ValueError(
                f"sampler row for {template.name} missing qid_{slot}: {row}"
            )
        anchors[slot] = {"qid": qid, "title": row.get(f"title_{slot}") or qid}
    return anchors


def filter_answers(
    answers: Sequence[str],
    anchor_qids: Sequence[str],
    max_answers: int = MAX_ANSWERS,
) -> list[str] | None:
    """Dedup answers, drop anchors from them; None if empty or > max_answers."""
    seen: set[str] = set(anchor_qids)
    out: list[str] = []
    for qid in answers:
        if qid and qid not in seen:
            seen.add(qid)
            out.append(qid)
    if not out or len(out) > max_answers:
        return None
    return out


def build_bridges(
    bridges: Sequence[str],
    anchor_qids: Sequence[str],
    answers: Sequence[str],
) -> list[str]:
    """Bridge set = intermediates + anchors, minus answers, deduped in order."""
    answer_set = set(answers)
    seen: set[str] = set()
    out: list[str] = []
    for qid in list(bridges) + list(anchor_qids):
        if qid and qid not in answer_set and qid not in seen:
            seen.add(qid)
            out.append(qid)
    return out


def relation_vocabulary(template: PatternTemplate) -> list[str]:
    """['P69 (educated at)', ...] for the verbalizer / round-trip judge."""
    return [f"{p} ({REL_LABELS.get(p, p)})" for p in template.relations]


# ---------------------------------------------------------------------------
# Neo4j access (guarded)
# ---------------------------------------------------------------------------

_driver = None


def _get_driver():
    global _driver
    if _driver is None:
        from neo4j import GraphDatabase  # lazy: keep module importable

        _driver = GraphDatabase.driver(
            config.NEO4J_URI, auth=config.NEO4J_AUTH, connection_timeout=5.0
        )
    return _driver


def neo4j_available() -> bool:
    """Cheap connectivity probe; never raises."""
    try:
        _get_driver().verify_connectivity()
        return True
    except Exception:
        return False


def run_cypher(query: str, params: dict | None = None) -> list[dict]:
    """Run a read query; raise Neo4jUnavailable with a skip message on failure."""
    try:
        driver = _get_driver()
        with driver.session() as session:
            return [r.data() for r in session.run(query, params or {})]
    except Neo4jUnavailable:
        raise
    except Exception as exc:  # ServiceUnavailable, auth, DNS, ...
        raise Neo4jUnavailable(
            f"neo4j not reachable at {config.NEO4J_URI} ({type(exc).__name__}: "
            f"{exc}) — start it with `docker compose up -d neo4j` and load "
            "Wikidata5M (ingest/load_neo4j.py); skipping KG-pattern generation."
        ) from exc


# ---------------------------------------------------------------------------
# Pipeline entry points
# ---------------------------------------------------------------------------


def sample_pattern(
    seed_qid: str | None = None,
    template: PatternTemplate | str | None = None,
    rng: random.Random | None = None,
    run_query: RunQuery | None = None,
) -> InstantiatedPattern | None:
    """Instantiate a template by sampling anchor entities from neo4j.

    seed_qid pins the template's primary anchor to a specific entity.
    Returns None when the sampler finds no binding (e.g. seed doesn't match).
    Raises Neo4jUnavailable when the graph is down and no run_query is given.
    """
    rng = rng or random.Random()
    if template is None:
        template = rng.choice(PATTERNS)
    elif isinstance(template, str):
        template = TEMPLATES_BY_NAME[template]
    q = run_query or run_cypher
    rows = q(template.sampler_cypher, {"seed": seed_qid})
    if not rows:
        return None
    return InstantiatedPattern(template=template, anchors=anchors_from_row(template, rows[0]))


def execute(
    pattern: InstantiatedPattern,
    run_query: RunQuery | None = None,
    max_answers: int = MAX_ANSWERS,
) -> tuple[list[str], list[str]] | None:
    """Run the pattern for its exact golden sets.

    Returns (answer_qids, bridge_qids), or None when the answer set is empty
    or larger than `max_answers` (filtered out per DESIGN.md).
    """
    q = run_query or run_cypher
    rows = q(pattern.template.exec_cypher, pattern.params)
    if not rows:
        return None
    raw_answers = rows[0].get("answers") or []
    raw_bridges = rows[0].get("bridges") or []
    answers = filter_answers(raw_answers, pattern.anchor_qids, max_answers)
    if answers is None:
        return None
    bridges = build_bridges(raw_bridges, pattern.anchor_qids, answers)
    return answers, bridges


def to_record(
    pattern: InstantiatedPattern,
    answers: list[str],
    bridges: list[str],
    question: str,
) -> QuestionRecord:
    return QuestionRecord(
        id=pattern.id,
        question=question,
        answer_qids=answers,
        bridge_qids=bridges,
        source="kg_pattern",
        cypher=pattern.cypher,
        difficulty=pattern.template.hops,
    )

"""Offline tests for the KG-pattern pipeline: template inventory sanity plus
sample/execute logic against an injected in-memory query runner (no neo4j)."""

import random

import pytest

from questiongen import kg_patterns as kp

ALLOWED_PIDS = set(kp.REL_LABELS)


# ---------------------------------------------------------------------------
# Template inventory
# ---------------------------------------------------------------------------


def test_inventory_hops_range_and_kinds():
    assert len(kp.PATTERNS) >= 8
    for t in kp.PATTERNS:
        assert 2 <= t.hops <= 4, t.name
        assert t.kind in {"chain", "star", "intersection"}, t.name
        assert len(t.relations) == t.hops, f"{t.name}: relations != hops"
    hops = {t.hops for t in kp.PATTERNS}
    assert hops == {2, 3, 4}
    kinds = {t.kind for t in kp.PATTERNS}
    assert {"chain", "star", "intersection"} <= kinds


def test_inventory_uses_only_declared_relations():
    for t in kp.PATTERNS:
        assert set(t.relations) <= ALLOWED_PIDS, t.name


def test_inventory_covers_the_required_pids():
    used = {p for t in kp.PATTERNS for p in t.relations}
    assert used == ALLOWED_PIDS  # every relation in the vocabulary is exercised


def test_templates_are_consistent():
    for t in kp.PATTERNS:
        # every anchor slot appears in sampler outputs, exec params, semantics
        for slot in t.anchors:
            assert f"qid_{slot}" in t.sampler_cypher, t.name
            assert f"title_{slot}" in t.sampler_cypher, t.name
            assert f"$qid_{slot}" in t.exec_cypher, t.name
            assert "{" + slot + "}" in t.semantics, t.name
        assert "$seed" in t.sampler_cypher, t.name
        assert "answers" in t.exec_cypher and "bridges" in t.exec_cypher, t.name
    names = [t.name for t in kp.PATTERNS]
    assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

T = kp.TEMPLATES_BY_NAME["chain_film_director_school"]
T2 = kp.TEMPLATES_BY_NAME["star_school_occupation"]


def test_fill_semantics_substitutes_titles():
    anchors = {"school": {"qid": "Q49108", "title": "MIT"}}
    s = kp.fill_semantics(T, anchors)
    assert '"MIT" (Q49108)' in s
    assert "{school}" not in s


def test_render_cypher_substitutes_longest_first():
    cy = "MATCH (a {qid: $qid_film_a})--(b {qid: $qid_film})"
    out = kp.render_cypher(cy, {"qid_film": "Q1", "qid_film_a": "Q2"})
    assert "$" not in out
    assert "'Q2'" in out and "'Q1'" in out


def test_anchors_from_row_missing_slot_raises():
    with pytest.raises(ValueError):
        kp.anchors_from_row(T, {"title_school": "MIT"})


def test_filter_answers_drops_empty_and_oversize():
    assert kp.filter_answers([], []) is None
    assert kp.filter_answers([f"Q{i}" for i in range(31)], []) is None
    assert kp.filter_answers([f"Q{i}" for i in range(30)], []) is not None


def test_filter_answers_dedupes_and_removes_anchors():
    got = kp.filter_answers(["Q1", "Q1", "Q9", "Q2"], anchor_qids=["Q9"])
    assert got == ["Q1", "Q2"]


def test_build_bridges_excludes_answers_and_dedupes():
    got = kp.build_bridges(
        bridges=["Q10", "Q10", "Q1"], anchor_qids=["Q99"], answers=["Q1"]
    )
    assert got == ["Q10", "Q99"]


def test_relation_vocabulary_labels():
    assert kp.relation_vocabulary(T) == ["P57 (director)", "P69 (educated at)"]


# ---------------------------------------------------------------------------
# sample_pattern / execute with an injected fake query runner
# ---------------------------------------------------------------------------


def fake_runner(sampler_rows, exec_rows):
    """Return a RunQuery that answers samplers and exec queries from fixtures."""

    def run(query: str, params: dict) -> list[dict]:
        if "rand()" in query:  # sampler queries randomize; exec queries don't
            seed = params.get("seed")
            if seed is not None:
                return [r for r in sampler_rows if seed in r.values()]
            return sampler_rows
        return exec_rows

    return run


SAMPLER_ROW = {"qid_school": "Q49108", "title_school": "MIT"}


def test_sample_pattern_binds_anchors():
    q = fake_runner([SAMPLER_ROW], [])
    pat = kp.sample_pattern(template=T, run_query=q, rng=random.Random(0))
    assert pat is not None
    assert pat.anchors["school"] == {"qid": "Q49108", "title": "MIT"}
    assert pat.params == {"qid_school": "Q49108"}
    assert "'Q49108'" in pat.cypher and "$qid_school" not in pat.cypher
    assert "MIT" in pat.semantics
    assert pat.id.startswith("kg-")


def test_sample_pattern_seed_filters():
    q = fake_runner([SAMPLER_ROW], [])
    assert kp.sample_pattern(seed_qid="Q49108", template=T, run_query=q) is not None
    assert kp.sample_pattern(seed_qid="Q999", template=T, run_query=q) is None


def test_sample_pattern_by_template_name():
    q = fake_runner([SAMPLER_ROW], [])
    pat = kp.sample_pattern(template="chain_film_director_school", run_query=q)
    assert pat is not None and pat.template is T


def test_execute_returns_golden_sets():
    q = fake_runner(
        [SAMPLER_ROW],
        [{"answers": ["Q1", "Q2", "Q2"], "bridges": ["Q10", "Q1"]}],
    )
    pat = kp.sample_pattern(template=T, run_query=q)
    result = kp.execute(pat, run_query=q)
    assert result is not None
    answers, bridges = result
    assert answers == ["Q1", "Q2"]  # deduped
    # bridge = intermediates + anchor, minus answers
    assert bridges == ["Q10", "Q49108"]


def test_execute_filters_empty_answer_set():
    q = fake_runner([SAMPLER_ROW], [{"answers": [], "bridges": ["Q10"]}])
    pat = kp.sample_pattern(template=T, run_query=q)
    assert kp.execute(pat, run_query=q) is None


def test_execute_filters_oversize_answer_set():
    big = [f"Q{i}" for i in range(1, 40)]
    q = fake_runner([SAMPLER_ROW], [{"answers": big, "bridges": []}])
    pat = kp.sample_pattern(template=T, run_query=q)
    assert kp.execute(pat, run_query=q) is None
    # boundary: exactly 30 answers survives
    q30 = fake_runner([SAMPLER_ROW], [{"answers": big[:30], "bridges": []}])
    assert kp.execute(pat, run_query=q30) is not None


def test_to_record_shape():
    q = fake_runner(
        [SAMPLER_ROW], [{"answers": ["Q1"], "bridges": ["Q10"]}]
    )
    pat = kp.sample_pattern(template=T, run_query=q)
    answers, bridges = kp.execute(pat, run_query=q)
    rec = kp.to_record(pat, answers, bridges, "Which films ...?")
    assert rec.source == "kg_pattern"
    assert rec.difficulty == T.hops == 2
    assert rec.answer_qids == ["Q1"]
    assert rec.bridge_qids == ["Q10", "Q49108"]
    assert rec.cypher and "'Q49108'" in rec.cypher


def test_two_anchor_template_flow():
    row = {
        "qid_school": "Q49108",
        "title_school": "MIT",
        "qid_occ": "Q82594",
        "title_occ": "computer scientist",
    }
    q = fake_runner([row], [{"answers": ["Q5", "Q6"], "bridges": []}])
    pat = kp.sample_pattern(template=T2, run_query=q)
    assert set(pat.params) == {"qid_school", "qid_occ"}
    answers, bridges = kp.execute(pat, run_query=q)
    assert answers == ["Q5", "Q6"]
    assert set(bridges) == {"Q49108", "Q82594"}  # anchors are bridge entities


def test_run_cypher_unreachable_raises_clear_message(monkeypatch):
    monkeypatch.setattr(kp.config, "NEO4J_URI", "bolt://127.0.0.1:1")

    class BoomDriver:
        def verify_connectivity(self):
            raise OSError("connection refused")

        def session(self):
            raise OSError("connection refused")

    monkeypatch.setattr(kp, "_get_driver", lambda: BoomDriver())
    assert kp.neo4j_available() is False
    with pytest.raises(kp.Neo4jUnavailable, match="skipping KG-pattern"):
        kp.run_cypher("RETURN 1", {})


# ---------------------------------------------------------------------------
# Anchor roles + title hygiene (full-run guards)
# ---------------------------------------------------------------------------


def test_templates_declare_roles_for_every_anchor():
    for t in kp.PATTERNS:
        assert len(t.roles) == len(t.anchors), t.name
        assert all(r for r in t.roles), t.name
    # every school slot uses the school role (with the yearbook exclusion)
    for t in kp.PATTERNS:
        for slot, role in zip(t.anchors, t.roles):
            if slot == "school":
                assert role is kp.ROLE_SCHOOL, t.name
    assert "yearbook" in kp.ROLE_SCHOOL  # the calibration failure classes
    assert "athletics" in kp.ROLE_SCHOOL
    assert "alumni" in kp.ROLE_SCHOOL
    assert "novel" in kp.ROLE_FILM


def test_title_ok_hygiene():
    assert kp.title_ok("MIT", "Q49108")
    assert not kp.title_ok("Sudden Death (Stephen Mertz novel)/Comments", "Q1")
    assert not kp.title_ok("Something#Fragment", "Q1")
    assert not kp.title_ok("", "Q1")
    assert not kp.title_ok(None, "Q1")
    assert not kp.title_ok("   ", "Q1")
    assert not kp.title_ok("Q1", "Q1")  # alias-less entity (title == QID)


def test_anchors_ok():
    good = kp.InstantiatedPattern(
        template=T, anchors={"school": {"qid": "Q1", "title": "MIT"}}
    )
    bad = kp.InstantiatedPattern(
        template=T, anchors={"school": {"qid": "Q1", "title": "Page/Comments"}}
    )
    assert kp.anchors_ok(good)
    assert not kp.anchors_ok(bad)

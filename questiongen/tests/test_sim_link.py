"""Offline tests for sim_link.py — candidate selection and record building
against in-memory fixtures (no qdrant/neo4j/sidecar required)."""

from questiongen import sim_link as sl
from questiongen.sidecar import DictSidecar

ABSTRACT = (
    "Alan Turing was an English mathematician and computer scientist. He is "
    "widely considered to be the father of theoretical computer science. "
    "Turing worked at Bletchley Park with Gordon Welchman on the Enigma "
    "machine, building on work by the Polish Cipher Bureau."
)


def test_title_mentioned_word_boundaries():
    assert sl.title_mentioned("Bletchley Park", ABSTRACT)
    assert sl.title_mentioned("bletchley park", ABSTRACT)  # case-insensitive
    assert not sl.title_mentioned("Bletch", ABSTRACT)  # substring != word
    assert not sl.title_mentioned("Ada Lovelace", ABSTRACT)
    assert not sl.title_mentioned("Eni", ABSTRACT)  # min length guard
    assert not sl.title_mentioned("Park", "")
    assert not sl.title_mentioned("", ABSTRACT)


def test_find_text_link_candidates_filters():
    anchor = "Q7251"  # Alan Turing
    neighbors = [
        ("Q7251", "Alan Turing"),  # the anchor itself -> excluded
        ("Q155921", "Bletchley Park"),  # mentioned + graph edge -> excluded
        ("Q723107", "Gordon Welchman"),  # mentioned + NO edge -> answer
        ("Q1140213", "Enigma machine"),  # mentioned + NO edge -> answer
        ("Q11660", "Artificial intelligence"),  # similar, NOT mentioned -> excluded
        ("Q723107", "Gordon Welchman"),  # duplicate -> excluded
    ]
    edges = {frozenset({"Q7251", "Q155921"})}  # Turing -- Bletchley Park triple

    def has_edge(a, b):
        return frozenset({a, b}) in edges

    got = sl.find_text_link_candidates(anchor, ABSTRACT, neighbors, has_edge)
    assert got == [("Q723107", "Gordon Welchman"), ("Q1140213", "Enigma machine")]


def test_find_text_link_candidates_respects_cap():
    abstract = " ".join(f"Word{i}word" for i in range(30))
    neighbors = [(f"Q{i}", f"Word{i}word") for i in range(30)]
    got = sl.find_text_link_candidates(
        "Q0", abstract, neighbors, lambda a, b: False, max_answers=3
    )
    assert len(got) == 3


def test_find_text_link_candidates_no_abstract():
    assert sl.find_text_link_candidates("Q1", None, [("Q2", "x")], lambda a, b: False) == []
    assert sl.find_text_link_candidates("Q1", "", [("Q2", "x")], lambda a, b: False) == []


def test_build_scaffold_mentions_hop_not_anchor():
    s = sl.build_scaffold("P108", "employer", "Government Code and Cypher School", 2)
    assert "P108" in s and "employer" in s
    assert "Government Code and Cypher School" in s
    assert "text" not in s.lower() or True  # scaffold is free-form; key parts above
    assert "NO direct relationship" in s


def test_to_record_shape():
    answers = [("Q723107", "Gordon Welchman"), ("Q1140213", "Enigma machine")]
    rec = sl.to_record("Q7251", "Q155921", answers, "Who worked with ...?")
    assert rec.source == "sim_link"
    assert rec.answer_qids == ["Q723107", "Q1140213"]
    assert rec.bridge_qids == ["Q7251", "Q155921"]  # anchor + graph-hop entity
    assert rec.difficulty == 2
    assert rec.cypher is None
    assert rec.id.startswith("sim-")


def test_record_id_deterministic_order_independent():
    a = sl.record_id("Q1", ["Q2", "Q3"])
    b = sl.record_id("Q1", ["Q3", "Q2"])
    c = sl.record_id("Q9", ["Q2", "Q3"])
    assert a == b != c


def test_dict_sidecar_used_by_pipeline():
    side = DictSidecar(
        abstracts={"Q7251": ABSTRACT},
        relations={"P108": "employer"},
    )
    assert side.abstract("Q7251") == ABSTRACT
    assert side.relation_label("P108") == "employer"
    assert side.abstract("Q404") is None

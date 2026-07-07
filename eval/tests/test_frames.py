"""Offline tests for eval/frames.py — parsing and title->QID mapping against
the DictSidecar fixture. The HF download runs only with ECS_FRAMES_NETWORK=1."""

import os

import pytest

from eval import frames
from questiongen.sidecar import DictSidecar


# ---------------------------------------------------------------------------
# wiki_links parsing
# ---------------------------------------------------------------------------


def test_parse_wiki_links_stringified_list():
    raw = "['https://en.wikipedia.org/wiki/Alan_Turing', 'https://en.wikipedia.org/wiki/Enigma_machine']"
    assert frames.parse_wiki_links(raw) == [
        "https://en.wikipedia.org/wiki/Alan_Turing",
        "https://en.wikipedia.org/wiki/Enigma_machine",
    ]


def test_parse_wiki_links_real_list_and_empty():
    assert frames.parse_wiki_links(["https://x/wiki/A", ""]) == ["https://x/wiki/A"]
    assert frames.parse_wiki_links(None) == []
    assert frames.parse_wiki_links("") == []
    assert frames.parse_wiki_links("[]") == []


def test_parse_wiki_links_falls_back_to_regex():
    raw = "see https://en.wikipedia.org/wiki/Alan_Turing and https://en.wikipedia.org/wiki/MIT"
    got = frames.parse_wiki_links(raw)
    assert got == [
        "https://en.wikipedia.org/wiki/Alan_Turing",
        "https://en.wikipedia.org/wiki/MIT",
    ]


def test_title_from_url():
    f = frames.title_from_url
    assert f("https://en.wikipedia.org/wiki/Alan_Turing") == "Alan Turing"
    assert f("https://en.wikipedia.org/wiki/Alan_Turing#Legacy") == "Alan Turing"
    assert f("https://en.wikipedia.org/wiki/G%C3%B6del") == "Gödel"
    assert f("https://en.wikipedia.org/wiki/A_(band)?x=1") == "A (band)"
    assert f("https://example.com/nope") is None


# ---------------------------------------------------------------------------
# row -> record mapping
# ---------------------------------------------------------------------------

SIDE = DictSidecar(
    titles={"Q7251": "Alan Turing"},
    aliases={
        "Alan Turing": ["Q7251"],
        "Enigma machine": ["Q1140213"],
        "Mercury": ["Q308", "Q925"],  # ambiguous alias: planet vs element
    },
)


def make_row(**kw) -> frames.FramesRow:
    base = dict(
        idx=3,
        question="Which machine did Alan Turing help break?",
        answer_text="Enigma",
        wiki_links=[
            "https://en.wikipedia.org/wiki/Alan_Turing",
            "https://en.wikipedia.org/wiki/Enigma_machine",
        ],
    )
    base.update(kw)
    return frames.FramesRow(**base)


def test_map_titles_to_qids():
    got = frames.map_titles_to_qids(["Alan Turing", "Enigma machine", "Nope"], SIDE)
    assert got == ["Q7251", "Q1140213"]


def test_map_titles_ambiguous_alias_is_deterministic():
    assert frames.map_titles_to_qids(["Mercury"], SIDE) == ["Q308"]  # lowest QID


def test_map_titles_dedupes():
    assert frames.map_titles_to_qids(["Alan Turing", "Alan Turing"], SIDE) == ["Q7251"]


def test_row_to_record_frames_contract():
    rec = frames.row_to_record(make_row(), SIDE)
    assert rec is not None
    assert rec.id == "frames-3"
    assert rec.source == "frames"
    # documented caveat: evidence articles go to answer_qids, bridge empty
    assert rec.answer_qids == ["Q7251", "Q1140213"]
    assert rec.bridge_qids == []
    assert rec.cypher is None
    assert rec.difficulty == 2  # proxy: number of linked articles


def test_row_to_record_unmappable_returns_none():
    row = make_row(wiki_links=["https://en.wikipedia.org/wiki/Unknown_Person"])
    assert frames.row_to_record(row, SIDE) is None
    assert frames.row_to_record(make_row(question=""), SIDE) is None


def test_row_from_raw_key_variants():
    row = frames.row_from_raw(
        0,
        {
            "Prompt": " q? ",
            "Answer": "a",
            "wiki_links": "['https://en.wikipedia.org/wiki/X_y']",
        },
    )
    assert row.question == "q?"
    assert row.titles == ["X y"]


# ---------------------------------------------------------------------------
# Live download (opt-in: network + HF)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("ECS_FRAMES_NETWORK") != "1",
    reason="set ECS_FRAMES_NETWORK=1 to test the HF download",
)
def test_live_download_and_parse():
    rows = frames.load_frames_rows("test")
    assert len(rows) > 500  # FRAMES has 824 rows
    with_links = [r for r in rows if r.wiki_links]
    assert len(with_links) > 500
    assert all(r.question for r in rows[:20])

"""Unit tests for the entity title rule (DESIGN.md "Entity titles")."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from build_sidecar import derive_title, prefix_title  # noqa: E402

HARVARD_ALIASES = ["Harvard geography", "Harvard University", "Harvard"]
HARVARD_ABSTRACT = (
    "Harvard University is a private Ivy League research university in "
    "Cambridge, Massachusetts."
)


def test_longest_prefix_matching_alias_wins():
    assert derive_title(HARVARD_ALIASES, HARVARD_ABSTRACT, "Q13371") == "Harvard University"


def test_case_insensitive_match():
    assert (
        derive_title(["harvard university"], HARVARD_ABSTRACT, "Q13371")
        == "harvard university"
    )


def test_leading_whitespace_and_quotes_stripped():
    abstract = '  "Harvard University is a private university."'
    assert derive_title(HARVARD_ALIASES, abstract, "Q13371") == "Harvard University"


def test_whitespace_normalized():
    assert (
        derive_title(["Harvard  University"], "Harvard University is old.", "Q1")
        == "Harvard  University"
    )
    # embedded runs of whitespace in the abstract are normalized too
    assert (
        derive_title(["Harvard University"], "Harvard   University is old.", "Q1")
        == "Harvard University"
    )


def test_fallback_to_first_alias_when_no_prefix_match():
    assert (
        derive_title(["Sport in Paris", "paname"], "Paris is the capital of France.", "Q90")
        == "Sport in Paris"
    )


def test_fallback_to_first_alias_when_no_abstract():
    assert derive_title(HARVARD_ALIASES, None, "Q13371") == "Harvard geography"
    assert derive_title(HARVARD_ALIASES, "", "Q13371") == "Harvard geography"


def test_final_fallback_to_qid():
    assert derive_title(None, HARVARD_ABSTRACT, "Q13371") == "Q13371"
    assert derive_title([], None, "Q13371") == "Q13371"


def test_prefix_title_none_when_no_match():
    assert prefix_title(["banana"], HARVARD_ABSTRACT) is None

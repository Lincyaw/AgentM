"""Witness validation: deterministic substring checks (design §4.b, §4.f).

Fail-stop position: every audit edge in v3 is gated on these rules; a
bug here corrupts every extracted graph downstream.
"""

from __future__ import annotations

from llmharness.audit.extractor.witness import normalize, witness_data, witness_ref


def test_normalize_is_case_and_whitespace_insensitive() -> None:
    """`"  HeLLo  World "` and `"hello world"` normalize equal."""

    assert normalize("  HeLLo  World ") == normalize("hello world")
    assert normalize("\tFoo\n\nBar  Baz") == "foo bar baz"




def test_witness_data_fails_when_entity_not_in_texts() -> None:
    """`bar` is in neither text → non-None error mentioning the entity."""

    err = witness_data(["bar"], src_text="alpha foo", dst_text="gamma foo")
    assert err is not None
    assert "bar" in err




def test_witness_ref_fails_when_quote_absent() -> None:
    """A quote that is not present produces a structured error."""

    err = witness_ref(
        "not present",
        src_text="alpha foo",
        dst_text="gamma foo",
    )
    assert err is not None
    assert "not present" in err

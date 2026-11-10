"""Degenerate-input parity: every case is compared against upstream jiwer 4.0.0.

The expected error classes below are those raised by *upstream*, not by this port.
Where upstream returns a value instead of raising, the value is asserted directly.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pytest

upstream = pytest.importorskip("jiwer")

import mojo_jiwer as mj

TOL = 1e-12


class Raising:
    """A transform whose output is a fixed object, used to break validation."""

    def __init__(self, value):
        self.value = value

    def __call__(self, sentence):
        return self.value


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reference", "hypothesis"),
    [
        ("", ""),
        ([], []),
        ([""], [""]),
        ([], [""]),
        ([""], []),
        (["", ""], ["", ""]),
    ],
)
@pytest.mark.parametrize("name", ["wer", "cer", "mer", "wil", "wip"])
def test_empty_input_scalar_measures_match_upstream(name, reference, hypothesis):
    # upstream measures.py wer/cer return `output.wer`, process.py process_words
    # sets `wer = num_insertions` (int 0) when num_rf_words == 0.
    ours = getattr(mj, name)(reference, hypothesis)
    theirs = getattr(upstream, name)(reference, hypothesis)
    assert ours == theirs
    assert type(ours) is type(theirs)


@pytest.mark.parametrize(
    ("reference", "hypothesis"),
    [("", ""), ([], []), ([""], [""]), ([], [""]), ([""], [])],
)
def test_empty_input_full_output_matches_upstream(reference, hypothesis):
    assert _as_dict(mj.process_words(reference, hypothesis)) == _as_dict(
        upstream.process_words(reference, hypothesis)
    )
    assert _as_dict(mj.process_characters(reference, hypothesis)) == _as_dict(
        upstream.process_characters(reference, hypothesis)
    )


def test_empty_reference_with_hypothesis_matches_upstream():
    # upstream process.py: `if num_rf_words == 0: wer = num_insertions`, and
    # `mer = 1, wip = 0` when the hypothesis is not empty.
    output = mj.process_words("", "hello world")
    reference = upstream.process_words("", "hello world")
    assert output.wer == reference.wer == 2
    assert output.mer == reference.mer == 1
    assert output.wip == reference.wip == 0
    assert output.wil == reference.wil == 1
    assert (output.hits, output.substitutions, output.insertions, output.deletions) == (
        reference.hits,
        reference.substitutions,
        reference.insertions,
        reference.deletions,
    )


def test_silence_match_preserves_information():
    # upstream process.py: num_rf_words == 0 and num_hp_words == 0 -> mer 0, wip 1.
    output = mj.process_words("", "")
    reference = upstream.process_words("", "")
    assert (output.wer, output.mer, output.wip, output.wil) == (
        reference.wer,
        reference.mer,
        reference.wip,
        reference.wil,
    )
    assert reference.wer == 0 and reference.wip == 1


# ---------------------------------------------------------------------------
# One element
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reference", "hypothesis"),
    [
        ("a", "a"),
        ("a", "b"),
        ("a", ""),
        ("", "a"),
        (" ", ""),
        ("é", "e"),
    ],
)
def test_single_element_input_matches_upstream(reference, hypothesis):
    for name in ["wer", "cer", "mer", "wil", "wip"]:
        ours = getattr(mj, name)(reference, hypothesis)
        theirs = getattr(upstream, name)(reference, hypothesis)
        assert ours == pytest.approx(theirs, abs=TOL)
    assert _as_dict(mj.process_words(reference, hypothesis)) == _as_dict(
        upstream.process_words(reference, hypothesis)
    )
    assert _as_dict(mj.process_characters(reference, hypothesis)) == _as_dict(
        upstream.process_characters(reference, hypothesis)
    )


def test_single_pair_multiword_word_runs_match_upstream():
    # one reference, one hypothesis, each a single token that repeats.
    reference = " ".join(["a"] * 5)
    hypothesis = " ".join(["b"] * 3)
    assert _as_dict(mj.process_words(reference, hypothesis)) == _as_dict(
        upstream.process_words(reference, hypothesis)
    )
    assert mj.wer(reference, hypothesis) == pytest.approx(
        upstream.wer(reference, hypothesis), abs=TOL
    )


# ---------------------------------------------------------------------------
# Duplicates / alignment tie-breaking (fixture values produced by upstream 4.0.0)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reference", "hypothesis", "counts", "chunks"),
    [
        ("a", "a a", (1, 0, 1, 0), [("equal", 0, 1, 0, 1), ("insert", 1, 1, 1, 2)]),
        (
            "a b",
            "b a",
            (1, 0, 1, 1),
            [
                ("insert", 0, 0, 0, 1),
                ("equal", 0, 1, 1, 2),
                ("delete", 1, 2, 2, 2),
            ],
        ),
        (
            "a b c",
            "b c d",
            (2, 0, 1, 1),
            [
                ("delete", 0, 1, 0, 0),
                ("equal", 1, 3, 0, 2),
                ("insert", 3, 3, 2, 3),
            ],
        ),
        (
            "a b c d",
            "a x y d",
            (2, 2, 0, 0),
            [
                ("equal", 0, 1, 0, 1),
                ("substitute", 1, 3, 1, 3),
                ("equal", 3, 4, 3, 4),
            ],
        ),
        (
            "the quick brown fox",
            "the brown quick fox",
            (3, 0, 1, 1),
            [
                ("equal", 0, 1, 0, 1),
                ("insert", 1, 1, 1, 2),
                ("equal", 1, 2, 2, 3),
                ("delete", 2, 3, 3, 3),
                ("equal", 3, 4, 3, 4),
            ],
        ),
    ],
)
def test_duplicate_tokens_use_upstream_tie_breaking(
    reference, hypothesis, counts, chunks
):
    output = mj.process_words(reference, hypothesis)
    expected = upstream.process_words(reference, hypothesis)
    assert (output.hits, output.substitutions, output.insertions, output.deletions) == (
        counts
    )
    assert (output.hits, output.substitutions, output.insertions, output.deletions) == (
        expected.hits,
        expected.substitutions,
        expected.insertions,
        expected.deletions,
    )
    assert [
        (chunk.type, chunk.ref_start_idx, chunk.ref_end_idx, chunk.hyp_start_idx, chunk.hyp_end_idx)
        for chunk in output.alignments[0]
    ] == chunks
    assert _as_dict(output) == _as_dict(expected)


@pytest.mark.parametrize(
    ("reference", "hypothesis"),
    [
        ("a a a b b", "a b b b"),
        ("x y x y", "y x y x"),
        ("a b a b a", "b a b a b"),
        ("aa aa aa", "aa aa"),
    ],
)
def test_repeated_token_alignments_match_upstream(reference, hypothesis):
    for name in ["wer", "mer", "wil", "wip"]:
        assert getattr(mj, name)(reference, hypothesis) == pytest.approx(
            getattr(upstream, name)(reference, hypothesis), abs=TOL
        )
    assert _as_dict(mj.process_words(reference, hypothesis)) == _as_dict(
        upstream.process_words(reference, hypothesis)
    )


# ---------------------------------------------------------------------------
# NaN / infinity: upstream rejects non-text input in the transform layer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("name", ["wer", "cer", "mer", "wil", "wip", "process_words"])
def test_non_finite_float_input_raises_upstream_error(name, bad):
    # upstream transforms.py AbstractTransform.__call__ raises ValueError for a
    # non str/list input.
    with pytest.raises(ValueError, match="expected to be a string or list"):
        getattr(mj, name)(bad, "a")
    with pytest.raises(ValueError, match="expected to be a string or list"):
        getattr(upstream, name)(bad, "a")
    with pytest.raises(ValueError, match="expected to be a string or list"):
        getattr(mj, name)("a", bad)


def test_non_finite_float_inside_list_raises_upstream_error_class():
    # upstream transforms.py Strip.process_string calls str.strip -> AttributeError.
    with pytest.raises(AttributeError):
        mj.cer([float("nan")], ["a"])
    with pytest.raises(AttributeError):
        upstream.cer([float("nan")], ["a"])
    # upstream wer_default starts with re.sub -> TypeError.
    with pytest.raises(TypeError):
        mj.wer([float("nan")], ["a"])
    with pytest.raises(TypeError):
        upstream.wer([float("nan")], ["a"])


def test_nan_word_strings_are_ordinary_tokens():
    # The strings "nan"/"inf" are words, not numbers; both implementations agree.
    for name in ["wer", "cer", "mer", "wil", "wip"]:
        assert getattr(mj, name)("nan inf", "nan") == pytest.approx(
            getattr(upstream, name)("nan inf", "nan"), abs=TOL
        )
    assert _as_dict(mj.process_words("nan inf", "nan")) == _as_dict(
        upstream.process_words("nan inf", "nan")
    )


# ---------------------------------------------------------------------------
# Mismatched shapes: same exception class and message as upstream
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reference", "hypothesis"),
    [
        (["one", "two"], ["one"]),
        (["one"], ["one", "two"]),
        (["one", "two", "three"], ["one"]),
        (["a", "b"], []),
        ([], ["a", "b"]),
    ],
)
@pytest.mark.parametrize("name", ["wer", "cer", "mer", "wil", "wip", "process_words"])
def test_mismatched_sentence_counts_raise_value_error(name, reference, hypothesis):
    # upstream process.py process_words raises ValueError when the transformed
    # reference and hypothesis counts differ.
    match = "their lengths must match"
    with pytest.raises(ValueError, match=match):
        getattr(mj, name)(reference, hypothesis)
    with pytest.raises(ValueError, match=match):
        getattr(upstream, name)(reference, hypothesis)


def test_mismatched_error_messages_are_identical():
    with pytest.raises(ValueError) as ours:
        mj.wer(["one", "two"], ["one"])
    with pytest.raises(ValueError) as theirs:
        upstream.wer(["one", "two"], ["one"])
    assert str(ours.value) == str(theirs.value)


@pytest.mark.parametrize("value", [{}, "", [[1]], "abc", [1, 2], [[None]]])
def test_invalid_transform_output_raises_upstream_error(value):
    # upstream process.py _apply_transform validates with
    # _is_list_of_list_of_strings and raises ValueError otherwise.
    transform = Raising(value)
    with pytest.raises(ValueError, match="list of strings"):
        mj.wer("a b", "a b", transform, transform)
    with pytest.raises(ValueError, match="list of strings"):
        upstream.wer("a b", "a b", transform, transform)


@pytest.mark.parametrize("value", [[], [["a"], ["b"]]])
def test_empty_and_unbalanced_transform_output_matches_upstream_value(value):
    # upstream process.py: `[]` is a valid list-of-lists-of-strings, and a
    # mismatched-but-valid output reaches process_words (here 0 references and
    # 0 hypotheses). The port must return exactly what upstream returns.
    transform = Raising(value)
    ours = mj.wer("a b", "a b", transform, transform)
    theirs = upstream.wer("a b", "a b", transform, transform)
    assert ours == theirs
    assert type(ours) is type(theirs)


@pytest.mark.parametrize("value", [None, 3, 2.5, b"abc", ("a",), {"a": "b"}])
def test_non_text_input_raises_upstream_error(value):
    # upstream transforms.py AbstractTransform.__call__ -> ValueError.
    with pytest.raises(ValueError, match="expected to be a string or list"):
        mj.wer(value, value)
    with pytest.raises(ValueError, match="expected to be a string or list"):
        upstream.wer(value, value)


# ---------------------------------------------------------------------------
# collect_error_counts on degenerate alignments
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reference", "hypothesis"),
    [("", ""), ([], []), ("a", "a"), ("a b", "b a"), ("a b c", "a x c")],
)
def test_collect_error_counts_degenerate_matches_upstream(reference, hypothesis):
    for ours, theirs in (
        (
            mj.collect_error_counts(mj.process_words(reference, hypothesis)),
            upstream.collect_error_counts(upstream.process_words(reference, hypothesis)),
        ),
        (
            mj.collect_error_counts(mj.process_characters(reference, hypothesis)),
            upstream.collect_error_counts(
                upstream.process_characters(reference, hypothesis)
            ),
        ),
    ):
        assert len(ours) == len(theirs) == 3
        assert all(isinstance(item, defaultdict) for item in ours)
        assert [dict(part) for part in ours] == [dict(part) for part in theirs]
        # upstream alignment.py collect_error_counts uses defaultdict(lambda: 0);
        # the contract is a fresh zero int for a missing key, not `int` identity.
        assert [bag.default_factory() for bag in ours] == [
            bag.default_factory() for bag in theirs
        ]
        assert all(type(bag.default_factory()) is int for bag in ours)
        assert [bag["never seen"] for bag in ours] == [
            bag["never seen"] for bag in theirs
        ]


# ---------------------------------------------------------------------------
# AlignmentChunk validation parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        ("unknown", 0, 1, 0, 1),
        ("equal", 2, 1, 0, 1),
        ("equal", 0, 1, 2, 1),
        ("substitute", 1, 3, 3, 1),
        (None, 0, 0, 0, 0),
    ],
)
def test_alignment_chunk_raises_same_error_as_upstream(args):
    # upstream process.py AlignmentChunk.__post_init__ raises ValueError.
    with pytest.raises(ValueError) as ours:
        mj.AlignmentChunk(*args)
    with pytest.raises(ValueError) as theirs:
        upstream.AlignmentChunk(*args)
    assert str(ours.value) == str(theirs.value)


def test_alignment_chunk_valid_values_match_upstream_semantics():
    for args in [("replace", 0, 1, 0, 1), ("equal", 0, 0, 0, 0), ("insert", 0, 0, 2, 4)]:
        ours = mj.AlignmentChunk(*args)
        theirs = upstream.AlignmentChunk(*args)
        assert (ours.type, ours.ref_start_idx, ours.ref_end_idx, ours.hyp_start_idx, ours.hyp_end_idx) == (
            theirs.type,
            theirs.ref_start_idx,
            theirs.ref_end_idx,
            theirs.hyp_start_idx,
            theirs.hyp_end_idx,
        )
    # upstream maps "replace" to "substitute" for rapidfuzz compatibility.
    assert mj.AlignmentChunk("replace", 0, 1, 0, 1).type == "substitute"


def _as_dict(output):
    from dataclasses import asdict

    return asdict(output)

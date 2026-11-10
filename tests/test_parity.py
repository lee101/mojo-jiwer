from __future__ import annotations

import inspect
import random
from collections import defaultdict
from dataclasses import asdict

import numpy as np
import pytest
from rapidfuzz.distance import Levenshtein

upstream = pytest.importorskip("jiwer")

import mojo_jiwer as mj
from mojo_jiwer import _native


def comparable(output):
    return asdict(output)


@pytest.mark.parametrize(
    ("reference", "hypothesis"),
    [
        ("", ""),
        ("", "background noise"),
        ("hello", ""),
        ("hello world", "hello duck"),
        ("hello world", "hello a b c world"),
        ("this is a test", "this was a difficult test"),
        ("  repeated   whitespace ", "repeated whitespace"),
        (
            ["one short sentence", "the second sentence is here"],
            ["one sentence", "the second phrase is here now"],
        ),
    ],
)
def test_word_measures_match_upstream(reference, hypothesis):
    assert mj.wer(reference, hypothesis) == upstream.wer(reference, hypothesis)
    assert mj.mer(reference, hypothesis) == upstream.mer(reference, hypothesis)
    assert mj.wil(reference, hypothesis) == upstream.wil(reference, hypothesis)
    assert mj.wip(reference, hypothesis) == upstream.wip(reference, hypothesis)


@pytest.mark.parametrize(
    ("reference", "hypothesis"),
    [
        ("", ""),
        ("", "abc"),
        ("abc", ""),
        ("kitten", "sitting"),
        ("naïve café", "naive cafe"),
        ("你好世界", "你号世"),
        ("a b", "ab"),
        (["first", "second"], ["frost", "seconds"]),
    ],
)
def test_cer_matches_upstream(reference, hypothesis):
    assert mj.cer(reference, hypothesis) == upstream.cer(reference, hypothesis)


@pytest.mark.parametrize(
    ("reference", "hypothesis"),
    [
        ("a", "a a"),
        ("a b", "b a"),
        ("a b", "b b b a"),
        ("a b c", "b c d"),
        ("a b c d", "a x y d"),
        ("the quick brown fox", "the brown quick fox"),
        (["short one here", "quite a bit longer"], ["shoe order one", "quite longer"]),
    ],
)
def test_process_words_full_output_matches_upstream(reference, hypothesis):
    assert comparable(mj.process_words(reference, hypothesis)) == comparable(
        upstream.process_words(reference, hypothesis)
    )


@pytest.mark.parametrize(
    ("reference", "hypothesis"),
    [
        ("abc", "axc"),
        ("aab", "baa"),
        ("résumé", "resume"),
        (["abc", "xyz"], ["ac", "xyzz"]),
    ],
)
def test_process_characters_full_output_matches_upstream(reference, hypothesis):
    assert comparable(mj.process_characters(reference, hypothesis)) == comparable(
        upstream.process_characters(reference, hypothesis)
    )


def test_random_alignment_and_counts_match_upstream():
    rng = random.Random(7)
    alphabet = ["a", "b", "c", "d"]
    for _ in range(300):
        reference = " ".join(rng.choices(alphabet, k=rng.randrange(0, 18)))
        hypothesis = " ".join(rng.choices(alphabet, k=rng.randrange(0, 18)))
        assert comparable(mj.process_words(reference, hypothesis)) == comparable(
            upstream.process_words(reference, hypothesis)
        )


def test_long_distance_uses_multiword_fallback():
    reference = " ".join(f"w{i % 13}" for i in range(180))
    hypothesis = " ".join(f"w{(i + (i % 7 == 0)) % 13}" for i in range(210))
    assert mj.wer(reference, hypothesis) == upstream.wer(reference, hypothesis)
    assert comparable(mj.process_words(reference, hypothesis)) == comparable(
        upstream.process_words(reference, hypothesis)
    )


@pytest.mark.parametrize("length", [64, 65, 127, 257, 1_000])
def test_multiword_distance_boundaries_and_simd_tail(length):
    reference = [index % 29 for index in range(length)]
    hypothesis = reference.copy()
    for index in range(3, length, 37):
        hypothesis[index] = 100 + index % 11
    hypothesis.insert(length // 3, 500)
    del hypothesis[(2 * length) // 3]
    assert _native.distance(reference, hypothesis) == Levenshtein.distance(
        reference, hypothesis
    )


def test_simd_prefix_and_suffix_tails_match_upstream():
    reference_tokens = list(range(30))
    hypothesis_tokens = reference_tokens.copy()
    hypothesis_tokens[15] = 100
    hypothesis_tokens[16] = 101
    assert _native.distance(reference_tokens, hypothesis_tokens) == 2

    reference = " ".join(f"w{token}" for token in reference_tokens)
    hypothesis = " ".join(f"w{token}" for token in hypothesis_tokens)
    assert mj.wer(reference, hypothesis) == upstream.wer(reference, hypothesis)
    assert comparable(mj.process_words(reference, hypothesis)) == comparable(
        upstream.process_words(reference, hypothesis)
    )


@pytest.mark.parametrize("count", [255, 256])
def test_distance_parallel_threshold_parity(count):
    reference = list(range(800))
    hypothesis = reference.copy()
    hypothesis[397] = 1000
    references = [reference] * count
    hypotheses = [hypothesis] * count
    assert _native.distances(references, hypotheses) == [1] * count


@pytest.mark.parametrize("count", [255, 256])
def test_trace_parallel_threshold_parity(count):
    reference = list(range(30))
    hypothesis = reference.copy()
    hypothesis[13:16] = [100, 101]
    expected = _native.trace(reference, hypothesis)
    assert _native.traces([reference] * count, [hypothesis] * count) == [
        expected
    ] * count


def test_custom_transform_parity():
    ours = mj.Compose(
        [
            mj.ToLowerCase(),
            mj.RemovePunctuation(),
            mj.ReduceToListOfListOfWords(),
        ]
    )
    theirs = upstream.Compose(
        [
            upstream.ToLowerCase(),
            upstream.RemovePunctuation(),
            upstream.ReduceToListOfListOfWords(),
        ]
    )
    reference = ["Hello, WORLD!", "Unicode: ¿Qué tal?"]
    hypothesis = ["hello world", "unicode qué mal"]
    assert mj.wer(reference, hypothesis, ours, ours) == upstream.wer(
        reference, hypothesis, theirs, theirs
    )


def test_contiguous_transform_parity():
    reference = ["one two", "three"]
    hypothesis = ["one", "two three"]
    assert mj.wer(
        reference,
        hypothesis,
        mj.wer_contiguous,
        mj.wer_contiguous,
    ) == upstream.wer(
        reference,
        hypothesis,
        upstream.wer_contiguous,
        upstream.wer_contiguous,
    )
    assert mj.cer(
        reference,
        hypothesis,
        mj.cer_contiguous,
        mj.cer_contiguous,
    ) == upstream.cer(
        reference,
        hypothesis,
        upstream.cer_contiguous,
        upstream.cer_contiguous,
    )


def test_collect_error_counts_matches_upstream():
    reference = ["the fast fox", "some words disappear"]
    hypothesis = ["the slow red fox", "some disappear now"]
    ours = mj.collect_error_counts(mj.process_words(reference, hypothesis))
    theirs = upstream.collect_error_counts(
        upstream.process_words(reference, hypothesis)
    )
    assert tuple(map(dict, ours)) == tuple(map(dict, theirs))


def test_mismatched_sentence_counts_raise_same_error():
    with pytest.raises(ValueError, match="their lengths must match"):
        mj.wer(["one", "two"], ["one"])


def test_invalid_transform_output_raises_same_error():
    with pytest.raises(ValueError, match="list of strings"):
        mj.wer("one", "one", lambda value: value, lambda value: value)


@pytest.mark.parametrize(
    "name", ["wer", "cer", "mer", "wil", "wip", "process_words", "process_characters"]
)
def test_public_parameter_names_match_upstream(name):
    assert list(inspect.signature(getattr(mj, name)).parameters) == list(
        inspect.signature(getattr(upstream, name)).parameters
    )


def test_alignment_chunk_validation_and_replace_alias():
    chunk = mj.AlignmentChunk("replace", 0, 1, 0, 1)
    assert chunk.type == "substitute"
    with pytest.raises(ValueError):
        mj.AlignmentChunk("unknown", 0, 1, 0, 1)
    with pytest.raises(ValueError, match="ref_start_idx"):
        mj.AlignmentChunk("equal", 2, 1, 0, 1)


def test_published_simple_vectors():
    assert mj.wer("hello world", "hello duck") == 0.5
    assert mj.wer("hello world", "hello a b c world") == 1.5
    assert mj.cer("abc", "axc") == pytest.approx(1 / 3)


TOLERANCE = 1e-12
"""Absolute tolerance for float comparisons against upstream jiwer 4.0.0.

The port computes the same rational error rates in different arithmetic orders,
so exact equality is not a meaningful contract; 1e-12 absorbs that without
weakening any comparison (the rates differ by at most a few ulps).
"""

SCALAR_GRID = [
    ("", ""),
    ("", "hello"),
    ("hello", ""),
    ("hello world", "hello duck"),
    ("hello world", "hello a b c world"),
    ("this is a test", "this was a difficult test"),
    ("a", "a"),
    ("a", "b"),
    ("a b c d e f", "a b x d e f"),
    (["one short sentence", "the second sentence is here"],
     ["one sentence", "the second phrase is here now"]),
    (["same", "words", "here"], ["same", "words", "here"]),
    (["", "x"], ["", "y"]),
    ("kitten sitting", "sitting kitten"),
    ("naïve café", "naive cafe"),
    ("你好世界", "你号世"),
    ("".join(chr(97 + i % 26) for i in range(400)),
     "".join(chr(97 + (i + (i % 11 == 0)) % 26) for i in range(430))),
]


@pytest.mark.parametrize(("reference", "hypothesis"), SCALAR_GRID)
@pytest.mark.parametrize("name", ["wer", "mer", "wil", "wip"])
def test_word_measures_match_upstream_exactly(name, reference, hypothesis):
    # No tolerance here: every rate on this grid is bit-identical to upstream.
    # A tolerance may hide a change in the arithmetic order, so parity is
    # asserted exactly and the tolerance below is kept only as a second net.
    ours = getattr(mj, name)(reference, hypothesis)
    theirs = getattr(upstream, name)(reference, hypothesis)
    assert ours == theirs
    assert type(ours) is type(theirs)


@pytest.mark.parametrize(("reference", "hypothesis"), SCALAR_GRID)
@pytest.mark.parametrize("name", ["wer", "mer", "wil", "wip"])
def test_word_measures_within_explicit_tolerance(name, reference, hypothesis):
    ours = getattr(mj, name)(reference, hypothesis)
    theirs = getattr(upstream, name)(reference, hypothesis)
    assert ours == pytest.approx(theirs, abs=TOLERANCE, rel=TOLERANCE)
    assert type(ours) is type(theirs)


@pytest.mark.parametrize(
    ("reference", "hypothesis"),
    [
        ("", ""),
        ("", "abc"),
        ("abc", ""),
        ("kitten", "sitting"),
        ("naïve café", "naive cafe"),
        ("你好世界", "你号世"),
        ("a b", "ab"),
        (["first", "second"], ["frost", "seconds"]),
        ("a" * 301, "a" * 240 + "b" * 61),
        (["", "abc", ""], ["a", "", "abc"]),
    ],
)
def test_cer_within_explicit_tolerance(reference, hypothesis):
    ours = mj.cer(reference, hypothesis)
    theirs = upstream.cer(reference, hypothesis)
    assert ours == pytest.approx(theirs, abs=TOLERANCE, rel=TOLERANCE)
    assert type(ours) is type(theirs)


@pytest.mark.parametrize(
    "name",
    ["wer", "cer", "mer", "wil", "wip"],
)
def test_measures_are_exactly_the_matching_output_fields(name):
    # upstream measures.py wer/cer return `output.wer`/`output.cer`; mer/wil/wip
    # return the same-named field of process_words (validated via the reference).
    reference = ["one two three", "four five"]
    hypothesis = ["one three", "four six seven"]
    output = mj.process_words(reference, hypothesis)
    characters = mj.process_characters(reference, hypothesis)
    expected = {
        "wer": output.wer,
        "mer": output.mer,
        "wil": output.wil,
        "wip": output.wip,
        "cer": characters.cer,
    }[name]
    assert getattr(mj, name)(reference, hypothesis) == pytest.approx(
        expected, abs=0.0
    )


BOUNDARY_LENGTHS = [63, 64, 65, 127, 257]
BOUNDARY_WER = {
    63: 0.06349206349206349,
    64: 0.0625,
    65: 0.06153846153846154,
    127: 0.047244094488188976,
    257: 0.03501945525291829,
}
BOUNDARY_COUNTS = {
    63: (60, 2, 1, 1),
    64: (61, 2, 1, 1),
    65: (62, 2, 1, 1),
    127: (122, 4, 1, 1),
    257: (249, 7, 1, 1),
}


def _boundary_pair(length):
    reference = [index % 29 for index in range(length)]
    hypothesis = reference.copy()
    for index in range(3, length, 37):
        hypothesis[index] = 100 + index % 11
    hypothesis.insert(length // 3, 500)
    del hypothesis[(2 * length) // 3]
    return reference, hypothesis


@pytest.mark.parametrize("length", BOUNDARY_LENGTHS)
def test_boundary_multiword_values_match_upstream_fixtures(length):
    # Expected values come from upstream jiwer 4.0.0 process_words on the same
    # word sequences, and are re-derived here to keep the fixture honest.
    # These lengths bracket the SIMD prefix/suffix and traceback boundaries; the
    # >= 64-word cases also run the multiword Myers kernel inside _native.distances.
    reference_ids, hypothesis_ids = _boundary_pair(length)
    reference = " ".join(f"w{token}" for token in reference_ids)
    hypothesis = " ".join(f"w{token}" for token in hypothesis_ids)
    output = mj.process_words(reference, hypothesis)
    expected = upstream.process_words(reference, hypothesis)
    counts = (output.hits, output.substitutions, output.insertions, output.deletions)
    assert counts == BOUNDARY_COUNTS[length]
    assert counts == (
        expected.hits,
        expected.substitutions,
        expected.insertions,
        expected.deletions,
    )
    assert output.wer == pytest.approx(BOUNDARY_WER[length], abs=TOLERANCE)
    assert output.wer == pytest.approx(expected.wer, abs=TOLERANCE)
    assert mj.wer(reference, hypothesis) == pytest.approx(
        upstream.wer(reference, hypothesis), abs=TOLERANCE
    )
    assert comparable(mj.process_words(reference, hypothesis)) == comparable(
        upstream.process_words(reference, hypothesis)
    )


def _process_grid():
    for name, reference, hypothesis in (
        ("a", "a", "a a"),
        ("b", "a b", "b a"),
        ("c", "a b c", "b c d"),
        ("d", "a b c d", "a x y d"),
        ("e", "the quick brown fox", "the brown quick fox"),
        ("f", ["short one here", "quite a bit longer"], ["shoe order one", "quite longer"]),
        ("g", ["", "a"], ["b", ""]),
        ("h", "", "a b c"),
        ("i", "a b c", ""),
        ("j", "x y z", "x z y"),
    ):
        yield name, reference, hypothesis


@pytest.mark.parametrize(
    ("name", "reference", "hypothesis"), list(_process_grid())
)
def test_process_float_fields_within_tolerance(name, reference, hypothesis):
    ours = comparable(mj.process_words(reference, hypothesis))
    theirs = comparable(upstream.process_words(reference, hypothesis))
    for field in ("wer", "mer", "wil", "wip"):
        assert ours[field] == theirs[field], f"{name}: {field}"
        assert ours[field] == pytest.approx(theirs[field], abs=TOLERANCE)
    for field in ("references", "hypotheses", "alignments",
                  "hits", "substitutions", "insertions", "deletions"):
        assert ours[field] == theirs[field], f"{name}: {field}"


@pytest.mark.parametrize(
    ("reference", "hypothesis"),
    [("abc", "axc"), ("aab", "baa"), ("résumé", "resume"),
     (["abc", "xyz"], ["ac", "xyzz"]), ("", "abc"), ("abc", ""),
     (["a b c", "d"], ["a b d", ""])],
)
def test_character_float_fields_within_tolerance(reference, hypothesis):
    ours = comparable(mj.process_characters(reference, hypothesis))
    theirs = comparable(upstream.process_characters(reference, hypothesis))
    assert ours["cer"] == theirs["cer"]
    assert ours["cer"] == pytest.approx(theirs["cer"], abs=TOLERANCE)
    for field in ("references", "hypotheses", "alignments",
                  "hits", "substitutions", "insertions", "deletions"):
        assert ours[field] == theirs[field]


def test_collect_error_counts_returns_upstream_types_and_values():
    reference = ["the fast fox", "some words disappear", "repeat repeat"]
    hypothesis = ["the slow red fox", "some disappear now", "repeat"]
    for process in ("process_words", "process_characters"):
        ours = mj.collect_error_counts(getattr(mj, process)(reference, hypothesis))
        theirs = upstream.collect_error_counts(
            getattr(upstream, process)(reference, hypothesis)
        )
        assert len(ours) == len(theirs) == 3
        for ours_bag, theirs_bag in zip(ours, theirs):
            assert isinstance(ours_bag, defaultdict)
            assert isinstance(theirs_bag, defaultdict)
            assert dict(ours_bag) == dict(theirs_bag)
            # upstream alignment.py uses defaultdict(lambda: 0).
            assert ours_bag.default_factory() == theirs_bag.default_factory() == 0
            assert type(ours_bag.default_factory()) is int


@pytest.mark.parametrize(
    "name",
    ["wer_default", "cer_default", "wer_contiguous", "cer_contiguous"],
)
def test_pipeline_defaults_come_from_upstream_transformation_spec(name):
    # upstream transformations.py builds these from Strip/RemoveMultipleSpaces/
    # ReduceToSingleSentence/ReduceToListOfList*/ReduceToSingleSentence.
    expected_types = [type(transform).__name__ for transform in getattr(upstream, name).transforms]
    ours = getattr(mj, name)
    assert [type(transform).__name__ for transform in ours.transforms] == expected_types
    sentences = ["  Hello,   WORLD!  ", "", "a b"]
    assert ours(sentences) == getattr(upstream, name)(sentences)


@pytest.mark.parametrize("count", [255, 256, 300])
def test_distance_batch_matches_upstream_against_shared_buffer(count):
    # Exercises the threaded and serial paths through the public WER API and
    # compares each sentence's distance with rapidfuzz (upstream's backend).
    rng = random.Random(count)
    references = []
    hypotheses = []
    for _ in range(count):
        reference = [rng.randrange(8) for _ in range(rng.randrange(0, 20))]
        hypothesis = [
            token if rng.random() > 0.2 else rng.randrange(8) for token in reference
        ]
        if rng.random() > 0.5:
            hypothesis.insert(rng.randrange(len(hypothesis) + 1), rng.randrange(8))
        references.append(reference)
        hypotheses.append(hypothesis)
    ours = _native.distances(references, hypotheses)
    assert ours == [
        Levenshtein.distance(reference, hypothesis)
        for reference, hypothesis in zip(references, hypotheses)
    ]


def test_native_flat_batch_rejects_unsafe_buffers():
    good = np.array([1, 2], dtype=np.int64)
    offsets = np.array([0, 2], dtype=np.int64)
    with pytest.raises(TypeError, match="dtype int64"):
        _native.distances_flat(good.astype(np.int32), good, offsets, offsets)
    with pytest.raises(ValueError, match="C-contiguous"):
        _native.distances_flat(good[::-1], good, offsets, offsets)
    with pytest.raises(ValueError, match="non-negative"):
        _native.distances_flat(np.array([-1], dtype=np.int64), good, np.array([0, 1]), offsets)
    with pytest.raises(ValueError, match="monotonically"):
        _native.distances_flat(good, good, np.array([0, 2, 1]), np.array([0, 1, 2]))
    with pytest.raises(ValueError, match="buffer length"):
        _native.distances_flat(good, good, np.array([0, 1]), np.array([0, 2]))


def test_native_batch_rejects_mismatched_pair_counts():
    with pytest.raises(ValueError, match="same number"):
        _native.distances([[1]], [])
    with pytest.raises(TypeError, match="must be integers"):
        _native.distance([1.5], [1])


def test_all_documented_core_transforms():
    assert mj.RemoveMultipleSpaces()("a   b") == "a b"
    assert mj.Strip()("  a  ") == "a"
    assert mj.ToLowerCase()("ÄBC") == "äbc"
    assert mj.ToUpperCase()("aß") == "ASS"
    assert mj.RemovePunctuation()("hi, ¿there?") == "hi there"
    assert mj.ReduceToListOfListOfWords()(["a b", "c"]) == [["a", "b"], ["c"]]
    assert mj.ReduceToListOfListOfChars()(["ab", "c"]) == [["a", "b"], ["c"]]
    assert mj.ReduceToSingleSentence()(["a", "", "b"]) == ["a b"]
    assert mj.Compose([mj.Strip(), mj.ToLowerCase()])(" A ") == "a"

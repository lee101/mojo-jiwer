from __future__ import annotations

import inspect
import random
from dataclasses import asdict

import numpy as np
import pytest

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

"""SIMD/vector tail coverage for the kernels in src/capi.mojo.

The vectorised loops in the shared library run `comptime W = simd_width_of[
DType.float64]()` wide on this machine, which is 8 lanes, so the runs that
matter are 0, 1, W-1, W, W+1 and a large odd length; each one that is not a
multiple of W is finished by the scalar tail loop. Every expectation here is
computed from upstream jiwer, not from this port, so a tail bug cannot hide.
"""

from __future__ import annotations

from collections import defaultdict

import pytest

upstream = pytest.importorskip("jiwer")

import mojo_jiwer as mj
from mojo_jiwer import _native

W = 16   # simd_width_of[DType.float64]() on this x86-64 build
LENGTHS = [0, 1, W - 1, W, W + 1, 1001]


def _ids(sequences: list[list[str]]) -> list[list[int]]:
    """Token lists as dense IDs, the way the Python wrapper hands them over."""
    ids = defaultdict()
    ids.default_factory = ids.__len__
    return [[ids[token] for token in sentence] for sentence in sequences]


def _ops_len(reference: list[str], hypothesis: list[str]) -> int:
    return len(_native.traces(_ids([reference]), _ids([hypothesis]))[0])


def _up_count(reference: list[str], hypothesis: list[str], name: str) -> int:
    return getattr(upstream.process_words(reference, hypothesis), name)


# ---------------------------------------------------------------------------
# matching_prefix / matching_suffix (vector scan + scalar tail)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("length", LENGTHS)
def test_identical_sequences_of_every_tail_length(length: int):
    """A full match leaves nothing for Myers, so the prefix scan alone decides."""
    tokens = [f"w{i}" for i in range(length)]
    result = mj.process_words(tokens, list(tokens))
    assert (result.hits, result.substitutions, result.deletions, result.insertions) == (
        length, 0, 0, 0,
    )
    assert _ops_len(tokens, tokens) == length


@pytest.mark.parametrize("length", LENGTHS)
def test_trailing_mismatch_exercises_suffix_scan(length: int):
    reference = [f"w{i}" for i in range(length)]
    hypothesis = list(reference)
    if length:
        hypothesis[-1] = "zzz"
    else:
        hypothesis = ["zzz"]
        reference = []
    result = mj.process_words(reference, hypothesis)
    sub = _up_count(reference, hypothesis, "substitutions")
    ins = _up_count(reference, hypothesis, "insertions")
    de = _up_count(reference, hypothesis, "deletions")
    assert (result.substitutions, result.insertions, result.deletions) == (sub, ins, de)


@pytest.mark.parametrize("length", LENGTHS)
def test_prefix_and_suffix_both_trimmed(length: int):
    """Both vector scans run, then Myers sees a short middle."""
    if length < 3:
        pytest.skip("needs a prefix, a middle and a suffix")
    reference = [f"a{i}" for i in range(length)]
    hypothesis = list(reference)
    hypothesis[length // 2] = "zzz"
    result = mj.process_words(reference, hypothesis)
    theirs = upstream.process_words(reference, hypothesis)
    assert (result.substitutions, result.insertions, result.deletions) == (
        theirs.substitutions, theirs.insertions, theirs.deletions,
    )
    ours_chunks = [
        (c.type, c.ref_start_idx, c.ref_end_idx, c.hyp_start_idx, c.hyp_end_idx)
        for c in result.alignments[0]
    ]
    theirs_chunks = [
        (c.type, c.ref_start_idx, c.ref_end_idx, c.hyp_start_idx, c.hyp_end_idx)
        for c in theirs.alignments[0]
    ]
    assert ours_chunks == theirs_chunks


# ---------------------------------------------------------------------------
# multiword Myers hint masks (two 64-bit words and up) and their reset
# ---------------------------------------------------------------------------


def _long_pair(length: int, seed: int = 0):
    base = [f"w{i}" for i in range(200)]
    reference = [base[(i * 7 + seed) % 200] for i in range(length)]
    hypothesis = list(reference)
    for i in range(0, length, 5):
        hypothesis[i] = base[(i * 11 + seed + 1) % 200]
    return reference, hypothesis


@pytest.mark.parametrize("length", [65, 66, 127, 128, 129, 1001])
def test_multiword_masks_without_repeats(length: int):
    """No token occurs twice among the first 64 rows, so every hint slot is
    written once and the one-pass build is exercised with nothing to OR."""
    reference = [f"u{i}" for i in range(length)]
    hypothesis = list(reference)
    hypothesis[length // 2] = "zzz"
    theirs = upstream.process_words(reference, hypothesis)
    ours = mj.process_words(reference, hypothesis)
    assert ours.substitutions == theirs.substitutions
    assert _ops_len(reference, hypothesis) == len(reference)


@pytest.mark.parametrize("length", [65, 300, 1001])
def test_multiword_repeated_tokens_are_or_all_bits(length: int):
    """The same token many times means the build must OR (~every) bit together;
    a fusion that drops the OR changes the distance, not just the speed."""
    reference = ["same"] * length
    hypothesis = ["same"] * length
    hypothesis[length // 2] = "other"
    theirs = upstream.process_words(reference, hypothesis)
    ours = mj.process_words(reference, hypothesis)
    assert (ours.substitutions, ours.insertions, ours.deletions) == (
        theirs.substitutions, theirs.insertions, theirs.deletions,
    )
    assert mj.wer(reference, hypothesis) == pytest.approx(theirs.wer, abs=1e-15)


@pytest.mark.parametrize("seed", range(4))
def test_multiword_batches_repeated_calls_are_stable(seed: int):
    """Distances and traces are expected to be a pure function of their input.

    A per-call reset that leaves a stale hint bit behind is only visible when
    the buffers are reused for a later pair, so call the same batch twice and
    check the second answer equals the first.
    """
    references, hypotheses = [], []
    for i in range(24):
        reference, hypothesis = _long_pair(80 + 17 * i, seed + i)
        references.append(reference)
        hypotheses.append(hypothesis)
    ref_ids, hyp_ids = _ids(references), _ids(hypotheses)
    first = _native.distances(ref_ids, hyp_ids)
    second = _native.distances(ref_ids, hyp_ids)
    assert first == second
    first_tr = _native.traces(ref_ids, hyp_ids)
    second_tr = _native.traces(ref_ids, hyp_ids)
    assert first_tr == second_tr
    for index, (reference, hypothesis, distance) in enumerate(
        zip(references, hypotheses, first)
    ):
        alone = _native.distances(
            [ref_ids[index]], [hyp_ids[index]]
        )[0]
        assert distance == alone


# ---------------------------------------------------------------------------
# Single-entry point (distance/reference_len == 0 paths and long pairs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("swapped", [False, True])
def test_single_pair_distance_parity(length: int, swapped: bool):
    # One string is one utterance; a list of words would be read as a list of
    # utterances by jiwer's length check.
    if length == 0:
        pytest.skip("upstream rejects an empty reference utterance")
    reference = " ".join(["a"] * length)
    hypothesis = " ".join("a" if i % 3 else "b" for i in range(length + 1))
    expected = upstream.process_words(reference, hypothesis)
    got = mj.process_words(reference, hypothesis)
    assert (got.substitutions, got.insertions, got.deletions) == (
        expected.substitutions, expected.insertions, expected.deletions,
    )
    ref_ids = [ord(t) for t in reference if t != " "]
    hyp_ids = [ord(t) for t in hypothesis if t != " "]
    arg = _native.distance(
        hyp_ids if swapped else ref_ids,
        ref_ids if swapped else hyp_ids,
    )
    assert arg == expected.substitutions + expected.insertions + expected.deletions

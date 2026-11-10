"""Serial and same-input concurrent batch calls must agree exactly.

The parallel phase kept the shipped serial kernels: in-kernel
`max.algorithm.parallelize` costs a fixed ~5.5-6 ms of launch overhead on this
toolchain, which is more than the kernel work at every size real callers pass,
so it was not added (see the README). This test pins the property that made
that safe to decide: a batch computed serially and the same batch computed by
several threads at once produce identical output. If an in-kernel worker path
is ever added, this is the test that catches drift between the two paths.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor

import pytest

upstream = pytest.importorskip("jiwer")

import mojo_jiwer as mj
from mojo_jiwer import _native

WORKERS = 4


def _corpus(count: int, length: int, seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    vocabulary = [f"word{i}" for i in range(64)]
    references, hypotheses = [], []
    for _ in range(count):
        reference = rng.choices(vocabulary, k=length)
        hypothesis = reference.copy()
        for index in range(0, length, 7):
            hypothesis[index] = rng.choice(vocabulary)
        references.append(" ".join(reference))
        hypotheses.append(" ".join(hypothesis))
    return references, hypotheses


def _encoded(*sentence_lists: list[str]) -> list[list[int]]:
    ids: dict[str, int] = {}
    return [
        [[ids.setdefault(token, len(ids)) for token in sentence.split()]
         for sentence in sentences]
        for sentences in sentence_lists
    ]


def _run_concurrently(fn):
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        return list(executor.map(lambda _: fn(), range(WORKERS)))


def _chunks(result):
    return [
        [
            (chunk.type, chunk.ref_start_idx, chunk.ref_end_idx,
             chunk.hyp_start_idx, chunk.hyp_end_idx)
            for chunk in sentence
        ]
        for sentence in result.alignments
    ]


@pytest.mark.parametrize("count,length", [(256, 12), (600, 20), (900, 8)])
def test_batched_distances_serial_and_threaded_match(count: int, length: int) -> None:
    references, hypotheses = _corpus(count, length, seed=count + length)
    serial = mj.wer(references, hypotheses)
    assert all(result == serial for result in
               _run_concurrently(lambda: mj.wer(references, hypotheses)))

    ref_ids, hyp_ids = _encoded(references, hypotheses)
    expected = _native.distances(ref_ids, hyp_ids)
    for distances in _run_concurrently(lambda: _native.distances(ref_ids, hyp_ids)):
        assert distances == expected


@pytest.mark.parametrize("count,length", [(256, 12), (600, 20)])
def test_batched_traces_serial_and_threaded_match(count: int, length: int) -> None:
    references, hypotheses = _corpus(count, length, seed=count * length)
    serial = mj.process_words(references, hypotheses)
    expected = upstream.process_words(references, hypotheses)
    assert serial.substitutions == expected.substitutions
    assert serial.deletions == expected.deletions
    assert serial.insertions == expected.insertions
    assert serial.hits == expected.hits
    assert _chunks(serial) == _chunks(expected)

    for result in _run_concurrently(lambda: mj.process_words(references, hypotheses)):
        assert result.substitutions == serial.substitutions
        assert result.deletions == serial.deletions
        assert result.insertions == serial.insertions
        assert result.hits == serial.hits
        assert _chunks(result) == _chunks(serial)
        assert result.wer == serial.wer

    ref_ids, hyp_ids = _encoded(references, hypotheses)
    serial_traces = _native.traces(ref_ids, hyp_ids)
    for traces in _run_concurrently(lambda: _native.traces(ref_ids, hyp_ids)):
        assert traces == serial_traces

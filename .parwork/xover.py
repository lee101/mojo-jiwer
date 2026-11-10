"""Crossover probe for the batch kernels.

Usage: python xover.py [worker-counts...]

For each case and batch size, times mji_distances / mji_traces with the batch
split into 1, 2, 4, ... chunks.  Chunks >1 here run on a Python ThreadPool so
the measurement is "does the work parallelise at all", independent of how the
workers are launched inside Mojo.
"""

from __future__ import annotations

import ctypes
import math
import os
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

I64 = ctypes.c_int64
HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    lib = ctypes.CDLL(path)
    lib.mji_distances.argtypes = [I64] * 12
    lib.mji_distances.restype = None
    lib.mji_traces.argtypes = [I64] * 12
    lib.mji_traces.restype = None
    return lib


def word_corpus(count, length, seed, vocab_size=256):
    rng = random.Random(seed)
    vocab = [f"word{i}" for i in range(vocab_size)]
    refs, hyps = [], []
    for _ in range(count):
        ref = [vocab.index(w) for w in rng.choices(vocab, k=length)]
        hyp = ref.copy()
        for i in range(0, length, 9):
            hyp[i] = rng.randrange(vocab_size)
        refs.append(ref)
        hyps.append(hyp)
    return refs, hyps


def flatten(refs, hyps):
    ref_offsets = np.zeros(len(refs) + 1, dtype=np.int64)
    hyp_offsets = np.zeros(len(hyps) + 1, dtype=np.int64)
    np.cumsum([len(r) for r in refs], out=ref_offsets[1:])
    np.cumsum([len(h) for h in hyps], out=hyp_offsets[1:])
    ref = np.array([t for r in refs for t in r] or [0], dtype=np.int64)
    hyp = np.array([t for h in hyps for t in h] or [0], dtype=np.int64)
    return ref, hyp, ref_offsets, hyp_offsets


def dist_blob(lib, refs, hyps, count, workers):
    ref, hyp, ro, ho = flatten(refs, hyps)
    ref_lengths = np.diff(ro)
    hyp_lengths = np.diff(ho)
    mws = max(1, (int(hyp_lengths.max(initial=0)) + 63) // 64)
    mso = (max(int(ref.max()), int(hyp.max())) + 1) * mws
    stride = mso + 2 * mws
    masks = np.zeros(max(1, stride * max(1, workers)), dtype=np.uint64)
    out = np.zeros(count, dtype=np.int64)

    def run(chunk):
        lib.mji_distances(
            ref.ctypes.data, hyp.ctypes.data, ro.ctypes.data, ho.ctypes.data,
            count, workers, chunk,
            masks.ctypes.data, stride if workers > 1 else 0,
            mws, mso, out.ctypes.data,
        )

    def go():
        if workers == 1:
            run(0)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(run, range(workers)))
    return go, out


def trace_blob(lib, refs, hyps, count, workers):
    ref, hyp, ro, ho = flatten(refs, hyps)
    ref_lengths = np.diff(ro)
    hyp_lengths = np.diff(ho)
    mstride = (int(ref_lengths.max(initial=0)) + 1) * (int(hyp_lengths.max(initial=0)) + 1)
    op_offsets = np.zeros(count + 1, dtype=np.int64)
    np.cumsum(ref_lengths + hyp_lengths, out=op_offsets[1:])
    ops = np.zeros(max(1, int(op_offsets[-1])), dtype=np.uint8)
    counts = np.zeros(count, dtype=np.int64)
    matrices = np.empty(mstride * max(1, workers), dtype=np.int32)

    def run(chunk):
        lib.mji_traces(
            ref.ctypes.data, hyp.ctypes.data, ro.ctypes.data, ho.ctypes.data,
            count, workers, chunk,
            matrices.ctypes.data, mstride if workers > 1 else 0,
            op_offsets.ctypes.data, ops.ctypes.data, counts.ctypes.data,
        )

    def go():
        if workers == 1:
            run(0)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(run, range(workers)))
    return go, (ops, op_offsets, counts)


def best(fn, out_check, repeats=7):
    best_time = math.inf
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        best_time = min(best_time, time.perf_counter() - t)
    return best_time


def main():
    workers = [int(x) for x in sys.argv[1:]] or [1, 2, 4, 8]
    lib = load(os.path.join(HERE, "baseline.so"))
    sizes = [int(x) for x in os.environ.get("SIZES", "0").split(",")] if os.environ.get("SIZES") else None
    cases = []
    for count, length in [(100, 12), (1000, 12), (10_000, 12), (2000, 60), (2000, 20)]:
        refs, hyps = word_corpus(count, length, count + length)
        cases.append((f"dist {count}x{length}w", "dist", refs, hyps))
        cases.append((f"trace {count}x{length}w", "trace", refs, hyps))
    for count, length in [(5000, 48)]:
        refs, hyps = word_corpus(count, length, 3, vocab_size=256)
        cases.append((f"dist {count}x{length}c", "dist", refs, hyps))

    print(f"lib={os.path.abspath(os.path.join(HERE, 'baseline.so'))} workers={workers}")
    print("| case | " + " | ".join(f"w={w}" for w in workers) + " |")
    print("|---|" + "---:|" * len(workers))
    for name, kind, refs, hyps in cases:
        count = len(refs)
        row = []
        base = None
        for w in workers:
            if w > count:
                row.append("-")
                continue
            builder = dist_blob if kind == "dist" else trace_blob
            go, out = builder(lib, refs, hyps, count, w)
            t = best(go, out)
            if base is None:
                base = t
            row.append(f"{t*1e3:.2f} ({base/t:.2f}x)")
        print(f"| {name} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()

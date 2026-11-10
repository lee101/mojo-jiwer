"""Re-measure the shipped serial batch kernels across sizes (wall + CPU)."""

import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, "/nvme0n1-disk/code/mojo-jiwer/python")
os.environ["MOJO_JIWER_LIB"] = "/nvme0n1-disk/code/mojo-jiwer/dist/libmojo-jiwer.so"
from mojo_jiwer import _native  # noqa: E402

import random  # noqa: E402


def word_corpus(count, length, seed, vocab_size=256):
    rng = random.Random(seed)
    vocab = list(range(vocab_size))
    R, H = [], []
    for _ in range(count):
        r = [rng.choice(vocab) for _ in range(length)]
        h = r.copy()
        for i in range(0, length, 9):
            h[i] = rng.choice(vocab)
        R.append(r)
        H.append(h)
    return R, H


def char_corpus(count, length, seed):
    rng = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyz "
    R, H = [], []
    for _ in range(count):
        r = [rng.choice(alphabet) for _ in range(length)]
        h = r.copy()
        for i in range(0, length, 11):
            h[i] = rng.choice(alphabet)
        R.append([ord(c) for c in r])
        H.append([ord(c) for c in h])
    return R, H


def measure(fn, repeat=5):
    wall, cpu = [], []
    for _ in range(repeat):
        w = time.perf_counter()
        c = time.process_time()
        fn()
        wall.append(time.perf_counter() - w)
        cpu.append(time.process_time() - c)
    return statistics.median(wall), statistics.median(cpu)


def flat(R, H):
    return _native._flatten_sequences(R, H)


cases = []
for count, length in [(1000, 12), (5000, 12), (10000, 12), (20000, 12), (5000, 20), (2000, 60), (5000, 60)]:
    R, H = word_corpus(count, length, count + length)
    cases.append((f"distance {count}x{length}w", "d", R, H))
for count in (5000, 10000, 20000, 40000):
    R, H = char_corpus(count, 48, count)
    cases.append((f"distance {count}x48c", "d", R, H))
for name, count, length in [("20w", 2000, 20), ("20w", 10000, 20), ("60w", 2000, 60)]:
    R, H = word_corpus(count, length, count)
    cases.append((f"trace {count}x{name}", "t", R, H))

print("| case | kernel wall (ms) | kernel cpu (ms) | tokens |")
print("|---|---:|---:|---:|")
for name, kind, R, H in cases:
    tokens = sum(len(r) for r in R) + sum(len(h) for h in H)
    ref, hyp, ro, ho = flat(R, H)
    if kind == "d":
        fn = lambda: _native.distances_flat(ref, hyp, ro, ho)
    else:
        fn = lambda: _native.traces(R, H)
    w, c = measure(fn)
    print(f"| {name} | {w*1e3:.2f} | {c*1e3:.2f} | {tokens} |", flush=True)

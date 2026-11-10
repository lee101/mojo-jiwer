"""Interleaved A/B crossover probe: serial (1 chunk) vs N chunks, CPU time.

Each measurement alternates serial and parallel on identical inputs, so host
load drifts affect both arms.  Reports the median of per-round ratios.
"""

from __future__ import annotations

import math
import os
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import xover


def cpu():
    return time.process_time()


def char_corpus(count, length, seed):
    rng = random.Random(seed)
    alpha = "abcdefghijklmnopqrstuvwxyz "
    R, H = [], []
    for _ in range(count):
        r = [rng.choice(alpha) for _ in range(length)]
        h = r.copy()
        for i in range(0, length, 11):
            h[i] = rng.choice(alpha)
        R.append([ord(c) for c in r])
        H.append([ord(c) for c in h])
    return R, H


def timeit(fn):
    t = cpu()
    fn()
    return cpu() - t


def main():
    lib = xover.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline.so"))
    workers = [int(x) for x in sys.argv[1:]] or [2, 3, 4, 8]
    rounds = int(os.environ.get("ROUNDS", "7"))
    cases = []
    if os.environ.get("CASES") == "trace":
        for c, l in [(1000, 12), (2000, 12), (5000, 12), (10000, 12), (2000, 20), (5000, 20), (10000, 20), (1000, 60), (2000, 60)]:
            R, H = xover.word_corpus(c, l, c + l)
            cases.append((f"trace {c}x{l}w", "trace", R, H))
    else:
        for c, l in [(5000, 12), (10000, 12), (20000, 12), (5000, 20), (10000, 20), (2000, 60), (5000, 60)]:
            R, H = xover.word_corpus(c, l, c + l)
            cases.append((f"dist {c}x{l}w", "dist", R, H))
        for c in (5000, 10000, 20000):
            R, H = char_corpus(c, 48, c)
            cases.append((f"dist {c}x48c", "dist", R, H))

    for name, kind, R, H in cases:
        count = len(R)
        builder = xover.dist_blob if kind == "dist" else xover.trace_blob
        go1, out1 = builder(lib, R, H, count, 1)
        go1()
        ref = out1.copy()
        cols = []
        for w in workers:
            gw, outw = builder(lib, R, H, count, w)
            gw()
            assert (outw == ref).all(), f"{name} w={w} mismatch"
            ratios, s, p = [], [], []
            for _ in range(rounds):
                ts = timeit(go1)
                tp = timeit(gw)
                s.append(ts)
                p.append(tp)
                ratios.append(ts / tp)
            cols.append(
                f"{statistics.median(s)*1e3:.2f}->{statistics.median(p)*1e3:.2f} "
                f"({statistics.median(ratios):.2f}x)"
            )
        print(f"{name:16s} " + "  ".join(f"w{w}: {c}" for w, c in zip(workers, cols)), flush=True)


if __name__ == "__main__":
    main()

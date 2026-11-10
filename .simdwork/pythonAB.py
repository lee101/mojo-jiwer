"""Interleaved A/B of the shipped bench cases through the *public* Python API.

Two shared libraries in one process, alternating repetitions, so drift from the
co-tenants on this 72-core box lands on both builds. Prints per-run arrays and
the median of ratios, which is what the README quotes.

    PYTHONPATH=python pythonAB.py OLD.so NEW.so [runs]
"""
import ctypes
import os
import statistics
import sys
import time

ROOT = "/nvme0n1-disk/code/mojo-jiwer"
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "bench"))

import bench as b  # noqa: E402

I64 = ctypes.c_int64


def load(path):
    lib = ctypes.CDLL(path)
    for name in ("mji_distances", "mji_traces"):
        f = getattr(lib, name)
        f.argtypes = [I64] * 12
        f.restype = None
    return lib


CASES = [
    ("WER: 10,000 x 12-word utterances", b.word_corpus(10_000, 12, 1), "wer"),
    ("WER: 2,000 x 60-word utterances", b.word_corpus(2_000, 60, 2), "wer"),
    ("CER: 5,000 x 48-character utterances", b.character_corpus(5_000, 48, 3), "cer"),
    ("process_words: 2,000 x 20 words", b.word_corpus(2_000, 20, 4), "process_words"),
    ("WER: one 1,000-word transcript", b.word_corpus(1, 1_000, 5), "wer"),
]


def timed(fn, reps):
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def key(value):
    if hasattr(value, "hits"):
        return (value.hits, value.substitutions, value.deletions, value.insertions)
    return value


def main():
    old_path, new_path = sys.argv[1], sys.argv[2]
    runs = int(sys.argv[3]) if len(sys.argv) > 3 else 9
    from mojo_jiwer import _native
    import mojo_jiwer

    libs = {"old": load(old_path), "new": load(new_path)}
    print(f"# {os.path.basename(old_path)}  vs  {os.path.basename(new_path)}")
    print("| case | run | old ms | new ms | old/new |")
    print("|---|---|---:|---:|---:|")
    ratios = {}
    for label, (refs, hyps), kind in CASES:
        fn = getattr(mojo_jiwer, kind)
        results = {}
        for side in ("old", "new"):
            _native._library = libs[side]
            results[side] = key(fn(refs, hyps))
        assert results["old"] == results["new"], f"{label}: results differ"
        per = {"old": [], "new": []}
        for r in range(runs):
            for side in ("old", "new"):
                _native._library = libs[side]
                per[side].append(timed(lambda: fn(refs, hyps), 3))
            print(f"| {label} | {r + 1} | {per['old'][-1] * 1e3:.2f} | "
                  f"{per['new'][-1] * 1e3:.2f} | {per['old'][-1] / per['new'][-1]:.3f} |")
        ratios[label] = [o / n for o, n in zip(per["old"], per["new"])]
    print()
    print("| case | median old/new | min | max |")
    print("|---|---:|---:|---:|")
    for label, rs in ratios.items():
        print(f"| {label} | {statistics.median(rs):.3f} | {min(rs):.3f} | {max(rs):.3f} |")


if __name__ == "__main__":
    main()

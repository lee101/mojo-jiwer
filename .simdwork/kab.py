"""Kernel-level A/B. Same process, alternating reps, same call the wrapper makes.

usage: kern_ab.py OLD.so NEW.so [loop]
loop: all | distances | traces | multiword
"""
import ctypes
import os
import sys
import time

ROOT = "/nvme0n1-disk/code/mojo-jiwer"
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "bench"))

import numpy as np

import bench as b  # noqa: E402
from mojo_jiwer import _native  # noqa: E402
from mojo_jiwer.process import _default_transform, _encode  # noqa: E402
from mojo_jiwer.transformations import wer_default, cer_default  # noqa: E402

I64 = ctypes.c_int64
A = _native._address


def load(path):
    lib = ctypes.CDLL(path)
    for name in ("mji_distances", "mji_traces"):
        f = getattr(lib, name)
        f.argtypes = [I64] * 12
        f.restype = None
    return lib


def masks_layout(rl, hl, ref, hyp, worker=1):
    short = int(min(rl.max(initial=0), hl.max(initial=0)))
    mws = max(1, (short + 63) // 64)
    swap = int(hl.max(initial=0)) > int(rl.max(initial=0))
    mso = ((int(hyp.max()) if swap else int(ref.max())) + 1) * mws
    return mws, mso, mso + 2 * mws


def make_dist(a, lib, worker=1):
    pairs = [(x, y) if len(y) <= len(x) else (y, x) for x, y in zip(a[0], a[1])]
    ref, hyp, ro, ho = _native._flatten_sequences([p[0] for p in pairs], [p[1] for p in pairs])
    rl, hl = np.diff(ro), np.diff(ho)
    count = len(ro) - 1
    mws, mso, stride = masks_layout(rl, hl, ref, hyp)
    masks = np.zeros(stride * worker, dtype=np.uint64)
    out = np.empty(count, dtype=np.int64)

    def run():
        lib.mji_distances(A(ref), A(hyp), A(ro), A(ho), count, worker, 0,
                          A(masks), stride if worker > 1 else 0, mws, mso, A(out))
    return run, out


def make_trace(a, lib):
    ref, hyp, ro, ho = _native._flatten_sequences(a[0], a[1])
    rl, hl = np.diff(ro), np.diff(ho)
    count = len(ro) - 1
    stride = (int(rl.max(initial=0)) + 1) * (int(hl.max(initial=0)) + 1)
    mats = np.empty(stride, dtype=np.int32)
    oo = np.empty(count + 1, dtype=np.int64)
    oo[0] = 0
    np.cumsum(rl + hl, out=oo[1:])
    ops = np.zeros(max(1, int(oo[-1])), dtype=np.uint8)
    cnt = np.empty(count, dtype=np.int64)

    def run():
        ops[:] = 0
        lib.mji_traces(A(ref), A(hyp), A(ro), A(ho), count, 1, 0, A(mats), 0,
                       A(oo), A(ops), A(cnt))
    return run, (ops, cnt)


def make_one_pair(pat_len, text_len, lib, seed=7):
    rng = np.random.default_rng(seed)
    pat = rng.integers(0, 253, pat_len, dtype=np.int64)
    txt = pat[:(text_len)]
    if text_len > pat_len:
        extra = rng.integers(0, 253, text_len - pat_len, dtype=np.int64)
        txt = np.concatenate([txt, extra])
    ro = np.array([0, pat_len], dtype=np.int64)
    ho = np.array([0, text_len], dtype=np.int64)
    rl = np.array([pat_len], dtype=np.int64)
    hl = np.array([text_len], dtype=np.int64)
    mws, mso, stride = masks_layout(rl, hl, pat, txt)
    masks = np.zeros(stride, dtype=np.uint64)
    out = np.empty(1, dtype=np.int64)

    def run():
        lib.mji_distances(A(pat), A(txt), A(ro), A(ho), 1, 1, 0, A(masks), 0,
                          mws, mso, A(out))
    return run, out


def interleaved(fa, fb, reps):
    ta = tb = float("inf")
    for _ in range(reps):
        t = time.perf_counter(); fa(); ta = min(ta, time.perf_counter() - t)
        t = time.perf_counter(); fb(); tb = min(tb, time.perf_counter() - t)
    return ta, tb


def prep(seed, count, length, kind):
    refs, hyps = (b.word_corpus(count, length, seed) if kind == "wer"
                  else b.character_corpus(count, length, seed))
    t = wer_default if kind == "wer" else cer_default
    return _encode(_default_transform(refs, t, True), _default_transform(hyps, t, False))


def flat(x):
    return [int(v) for arr in (x if isinstance(x, tuple) else (x,)) for v in arr.tolist()]


def main():
    old, new = load(sys.argv[1]), load(sys.argv[2])
    what = sys.argv[3] if len(sys.argv) > 3 else "all"
    print("| case | loop | before ms | after ms | after/before |")
    print("|---|---|---:|---:|---:|")
    cases = [
        ("WER 10k x 12w", (1, 10_000, 12, "wer"), "distances"),
        ("WER 2k x 60w", (2, 2_000, 60, "wer"), "distances"),
        ("CER 5k x 48c", (3, 5_000, 48, "cer"), "distances"),
        ("_native.distances 2k x 20w", (4, 2_000, 20, "wer"), "distances"),
        ("WER 1 x 1000w", (5, 1, 1_000, "wer"), "distances"),
        ("traces 2k x 20w", (4, 2_000, 20, "wer"), "traces"),
    ]
    for label, spec, loop in cases:
        if what not in ("all", loop):
            continue
        a = prep(*spec)
        make = make_dist if loop == "distances" else make_trace
        fa, ca = make(a, old)
        fb, cb = make(a, new)
        fa(); fb()
        assert flat(ca) == flat(cb), f"{label}: output changed"
        reps = 9 if "traces" in label else 25
        ta, tb = interleaved(fa, fb, reps)
        print(f"| {label} | {loop} | {ta*1e3:.3f} | {tb*1e3:.3f} | {tb/ta:.3f} |")

    # Multiword mask traffic: one call, text len 1, growing 1-word pattern.
    print()
    print("| myers_multiword mask work (text=1, 1 word) | pat len | before us | after us | after/before |")
    print("|---|---:|---:|---:|---:|")
    for plen in (65, 200, 600, 1000):
        fa, ca = make_one_pair(plen, 1, old)
        fb, cb = make_one_pair(plen, 1, new)
        fa(); fb()
        assert flat(ca) == flat(cb)
        ta, tb = interleaved(fa, fb, 200)
        print(f"| mask set+reset | {plen} | {ta*1e6:.1f} | {tb*1e6:.1f} | {tb/ta:.3f} |")


if __name__ == "__main__":
    main()

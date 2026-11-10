"""Parity + timing for kernel2 (exact serial kernel lifted verbatim)."""

from __future__ import annotations

import ctypes
import os
import random
import statistics
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
lib = ctypes.CDLL(os.path.join(HERE, "kernel2.so"))
for name in ("k_serial", "k_parallel"):
    fn = getattr(lib, name)
    fn.argtypes = [ctypes.c_int64] * 10
    fn.restype = ctypes.c_int64


def corpus(count, lo, hi, seed):
    rng = random.Random(seed)
    R, H = [], []
    for _ in range(count):
        L = rng.randint(lo, hi)
        r = [rng.randrange(256) for _ in range(L)]
        h = r.copy()
        for i in range(0, L, 9):
            h[i] = rng.randrange(256)
        R.append(r)
        H.append(h)
    return R, H


def prep(R, H, workers):
    ro = np.zeros(len(R) + 1, dtype=np.int64)
    ho = np.zeros(len(H) + 1, dtype=np.int64)
    np.cumsum([len(r) for r in R], out=ro[1:])
    np.cumsum([len(h) for h in H], out=ho[1:])
    ref = np.array([t for r in R for t in r] or [0], dtype=np.int64)
    hyp = np.array([t for h in H for t in h] or [0], dtype=np.int64)
    mws = max(1, (int(np.diff(ho).max(initial=0)) + 63) // 64)
    mso = (max(int(ref.max()), int(hyp.max())) + 1) * mws
    stride = mso + 2 * mws
    masks = np.zeros(stride * max(1, workers), dtype=np.uint64)
    res = np.zeros(len(R), dtype=np.int64)
    return ref, hyp, ro, ho, masks, res, stride, mso


def call(fn, key, masks_slot, workers):
    ref, hyp, ro, ho, masks, res, stride, mso = key
    fn(
        ref.ctypes.data, hyp.ctypes.data, ro.ctypes.data, ho.ctypes.data,
        res.ctypes.data, masks.ctypes.data + masks_slot * stride * 8, stride, mso,
        len(ro) - 1, workers,
    )
    return res.copy()


def timed(fn, key, slot, workers):
    t = time.perf_counter()
    fn(
        key[0].ctypes.data, key[1].ctypes.data, key[2].ctypes.data, key[3].ctypes.data,
        key[5].ctypes.data, key[4].ctypes.data + slot * key[6] * 8, key[6], key[7],
        len(key[2]) - 1, workers,
    )
    return time.perf_counter() - t


if __name__ == "__main__":
    for label, count, lo, hi in [
        ("2000 x 6-40w", 2000, 6, 40),
        ("10000 x 6-40w", 10000, 6, 40),
        ("50000 x 6-40w", 50000, 6, 40),
        ("2000 x 20w", 2000, 20, 20),
    ]:
        R, H = corpus(count, lo, hi, hash(label) & 0xFFFF)
        serial = prep(R, H, 1)
        expected = call(lib.k_serial, serial, 0, 1)
        row = []
        for w in (2, 3, 4, 8):
            par = prep(R, H, w)
            got = call(lib.k_parallel, par, 0, w)
            assert (got == expected).all(), f"{label} w={w} mismatch"
            row.append(w)
        # interleaved timing, best of 5
        ratios = []
        for _ in range(5):
            ts = min(timed(lib.k_serial, serial, 0, 1) for _ in range(3))
            line = [f"serial {ts*1e3:.2f}"]
            for w in row:
                par = prep(R, H, w)
                tp = min(timed(lib.k_parallel, par, 0, w) for _ in range(3))
                ratios.append(ts / tp)
                line.append(f"w{w} {tp*1e3:.2f} ({ts/tp:.2f}x)")
            print(label, " | ".join(line))
        print(label, "median ratio", f"{statistics.median(ratios):.2f}x")

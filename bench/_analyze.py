"""ANALYZE phase: one-shot measurement script. Holds the bench lock once.

Every number written to bench/analysis_raw.json is measured in this run.
Run:  flock /tmp/mojo-bench.lock pixi run python -u bench/_analyze.py
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import sys
import time
from itertools import chain

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.join(REPO, "bench"))

import numpy as np

import bench as corpus
import jiwer as upstream
import mojo_jiwer as mojo
from mojo_jiwer import _native
from mojo_jiwer.process import _chunks, _default_transform, _encode
from mojo_jiwer.transformations import cer_default, wer_default

OUT = {}

W10k = corpus.word_corpus(10_000, 12, 1)
W2k60 = corpus.word_corpus(2_000, 60, 2)
C5k = corpus.character_corpus(5_000, 48, 3)
W2k20 = corpus.word_corpus(2_000, 20, 4)
W1k = corpus.word_corpus(1, 1_000, 5)
# small sizes that real callers also pass
S1 = corpus.word_corpus(1, 20, 11)
S2 = corpus.word_corpus(100, 12, 12)


def best(fn, repeat=5):
    b = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        b = min(b, time.perf_counter() - t)
    return b


def now():
    return time.perf_counter()


# ---------------------------------------------------------------- baselines
def py_baseline(tokens_ref, tokens_hyp):
    """Pure-Python DP distance for one pair (the baseline upstream idea)."""
    n, m = len(tokens_ref), len(tokens_hyp)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        tr = tokens_ref[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (tr != tokens_hyp[j - 1]))
        prev = cur
    return prev[m]


def npy_dp_pairs(pairs):
    """Vectorized-by-row numpy DP (myers alternative baseline)."""
    total = 0
    for r, h in pairs:
        n, m = len(r), len(h)
        prev = np.arange(m + 1, dtype=np.int64)
        hh = np.asarray(h, dtype=np.int64)
        for i in range(1, n + 1):
            cur = np.empty(m + 1, dtype=np.int64)
            cur[0] = i
            np.minimum(prev[1:] + 1, prev[:-1] + (hh != r[i - 1]), out=cur[1:])
            cur[1:] = np.minimum(cur[1:], cur[:-1] + 1)
            prev = cur
        total += int(prev[-1])
    return total


# ---------------------------------------------------------------- helpers
def pairs_of(ids_ref, ids_hyp):
    return [(r, h) if len(h) <= len(r) else (h, r) for r, h in zip(ids_ref, ids_hyp)]


def flat(pairs):
    return _native._flatten_sequences([p[0] for p in pairs], [p[1] for p in pairs])


def distance_kernel_serial(pairs, repeat=3):
    fr, fh, fro, fho = flat(pairs)
    rl, hl = np.diff(fro), np.diff(fho)
    mx = int(np.maximum(rl, hl).max(initial=0))
    words = max(1, (mx + 63) // 64)
    top = max(int(fr.max()), int(fh.max())) + 1
    masks = np.zeros(top * words + 2 * words, dtype=np.uint64)
    res = np.empty(len(fro) - 1, dtype=np.int64)
    lib = _native.library()
    a = [("r", _native._address(x)) for x in (fr, fh, fro, fho, masks, res)]
    ca, ch, cro, cho, cm, cr = (v for _, v in a)
    n = len(fro) - 1

    def call():
        lib.mji_distances(ca, ch, cro, cho, n, 1, 0, cm, 0, words, top * words, cr)

    t = best(call, repeat)
    return t, n


def trace_kernel_serial(pairs, repeat=3):
    fr, fh, fro, fho = flat(pairs)
    rl, hl = np.diff(fro), np.diff(fho)
    stride = (int(rl.max(initial=0)) + 1) * (int(hl.max(initial=0)) + 1)
    mats = np.empty(stride, dtype=np.int32)
    op_off = np.empty(len(fro), dtype=np.int64)
    op_off[0] = 0
    np.cumsum(rl + hl, out=op_off[1:])
    ops = np.empty(max(1, int(op_off[-1])), dtype=np.uint8)
    counts = np.empty(len(fro) - 1, dtype=np.int64)
    lib = _native.library()
    n = len(fro) - 1
    ar = lambda x: _native._address(x)  # noqa: E731
    ca, ch, cro, cho, cm, co, cop, cc = map(ar, (fr, fh, fro, fho, mats, op_off, ops, counts))

    def call():
        lib.mji_traces(ca, ch, cro, cho, n, 1, 0, cm, 0, co, cop, cc)

    return best(call, repeat), n, int(op_off[-1]), stride


# ---------------------------------------------------------------- main
def main():
    OUT["machine"] = corpus.cpu_name()
    OUT["numpy"] = np.__version__
    OUT["python"] = sys.version.split()[0]
    OUT["loadavg_at_start"] = os.getloadavg()
    cases = []

    def run_case(label, ours, theirs, refs, hyps, kind):
        rows = {"label": label, "kind": kind, "n_pairs": len(refs)}
        rows["ours_s"] = best(lambda: ours(refs, hyps))
        rows["theirs_s"] = best(lambda: theirs(refs, hyps))
        # stage split
        tr = wer_default if kind != "cer" else cer_default
        is_ref = True
        rows["transform_s"] = best(
            lambda: (_default_transform(refs, tr, True), _default_transform(hyps, tr, False))
        )
        r_tok = _default_transform(refs, tr, True)
        h_tok = _default_transform(hyps, tr, False)
        rows["encode_s"] = best(lambda: _encode(r_tok, h_tok))
        if kind == "cer":
            r_txt = "".join(map(str, chain.from_iterable(r_tok)))
            rows["to_i64_s"] = best(
                lambda: (
                    np.frombuffer(
                        "".join(map(str, chain.from_iterable(r_tok))).encode("utf-32le"), dtype="<u4"
                    ).astype(np.int64),
                    np.frombuffer(
                        "".join(map(str, chain.from_iterable(h_tok))).encode("utf-32le"), dtype="<u4"
                    ).astype(np.int64),
                )
            )
            ref_off = np.empty(len(refs) + 1, dtype=np.int64)
            hyp_off = np.empty(len(hyps) + 1, dtype=np.int64)
            ref_off[0] = hyp_off[0] = 0
            # offsets are over characters, not utterances: sum token lengths
            np.cumsum(
                np.fromiter(map(len, r_tok), dtype=np.int64, count=len(r_tok)), out=ref_off[1:]
            )
            np.cumsum(
                np.fromiter(map(len, h_tok), dtype=np.int64, count=len(h_tok)), out=hyp_off[1:]
            )
            rd = np.frombuffer(r_txt.encode("utf-32le"), dtype="<u4").astype(np.int64)
            hd = np.frombuffer("".join(map(str, chain.from_iterable(h_tok))).encode("utf-32le"), dtype="<u4").astype(np.int64)
            # kernel inputs are dense IDs, as in production cer
            pairs = pairs_of(*_encode(r_tok, h_tok))
            rows["distances_flat_s"] = best(lambda: _native.distances_flat(rd, hd, ref_off, hyp_off))
            kt, n = distance_kernel_serial([(np.asarray(p[0], np.int64), np.asarray(p[1], np.int64)) for p in pairs])
            rows["dist_kernel_s"] = kt
            # tokens touched: sum of both sides of every pair
            rows["tokens"] = int(sum(len(p[0]) + len(p[1]) for p in pairs))
            rows["max_len"] = max(max(len(p[0]), len(p[1])) for p in pairs)
            rows["n_pairs"] = n
            rows["bytes"] = rows["tokens"] * 8
            rows["flops"] = rows["tokens"]  # one Myers step per text token
        else:
            pairs = pairs_of(_encode(r_tok, h_tok)[0], _encode(r_tok, h_tok)[1])
            rows["_native_distances_s"] = best(lambda: _native.distances(pairs_ref := [[int(x) for x in p[0]] for p in pairs], [[int(x) for x in p[1]] for p in pairs]), repeat=3)
            kt, n = distance_kernel_serial([(np.asarray(p[0], np.int64), np.asarray(p[1], np.int64)) for p in pairs])
            rows["dist_kernel_s"] = kt
            if kind == "wer":
                rows["tokens"] = int(sum(len(p[0]) + len(p[1]) for p in pairs))
                rows["max_len"] = max(max(len(p[0]), len(p[1])) for p in pairs)
                rows["n_pairs"] = n
                rows["bytes"] = rows["tokens"] * 8
                rows["flops"] = rows["tokens"]
                # python/numpy baselines on a small slice so they finish
                slice_pairs = pairs[:50]
                rows["npy_dp_50pairs_s"] = best(lambda: npy_dp_pairs(slice_pairs), repeat=2)
                one = pairs[0]
                rows["py_dp_1pair_s"] = best(lambda: py_baseline(list(one[0]), list(one[1])), repeat=3)
        if kind == "words":
            a = _encode(r_tok, h_tok)
            pr = pairs_of(a[0], a[1])
            tt, n, nops, stride = trace_kernel_serial([(np.asarray(p[0], np.int64), np.asarray(p[1], np.int64)) for p in pr])
            rows["trace_kernel_s"] = tt
            rows["traces_s"] = best(lambda: _native.traces(a[0], a[1]), repeat=3)
            rows["n_pairs"] = n
            rows["tokens"] = int(sum(len(p[0]) + len(p[1]) for p in pr))
            rows["max_len"] = max(max(len(p[0]), len(p[1])) for p in pr)
            cells = int(((np.diff(flat(pr)[2]) + 1) * (np.diff(flat(pr)[3]) + 1)).sum())
            rows["dp_cells"] = cells
            rows["trace_bytes_matrix"] = cells * 4
            rows["trace_bytes_ops"] = nops
            rows["trace_flops"] = 3 * cells
            traces_ops = _native.traces(a[0], a[1])
            rows["chunks_s"] = best(lambda: [_chunks(t) for t in traces_ops])
            rows["chunk_count"] = sum(len(_chunks(t)) for t in traces_ops)

            def reduce_loop():
                hits = subs = dels = ins = 0
                for rr, hh, ops in zip(a[0], a[1], traces_ops):
                    for c in _chunks(ops):
                        if c.type == "equal": hits += c.ref_end_idx - c.ref_start_idx
                        elif c.type == "substitute": subs += c.ref_end_idx - c.ref_start_idx
                        elif c.type == "delete": dels += c.ref_end_idx - c.ref_start_idx
                        else: ins += c.hyp_end_idx - c.hyp_start_idx
                return hits, subs, dels, ins

            rows["reduce_s"] = best(reduce_loop)
        cases.append(rows)
        OUT["cases"] = cases
        print(f"done {label}: ours={rows['ours_s']*1e3:.2f} upstream={rows['theirs_s']*1e3:.2f}", flush=True)

    run_case("wer_10k_x_12w", mojo.wer, upstream.wer, W10k[0], W10k[1], "wer")
    run_case("wer_2k_x_60w", mojo.wer, upstream.wer, W2k60[0], W2k60[1], "wer")
    run_case("cer_5k_x_48c", mojo.cer, upstream.cer, C5k[0], C5k[1], "cer")
    run_case("process_words_2k_x_20w", mojo.process_words, upstream.process_words, W2k20[0], W2k20[1], "words")
    run_case("wer_1_x_1000w", mojo.wer, upstream.wer, W1k[0], W1k[1], "wer")
    run_case("wer_1_x_20w", mojo.wer, upstream.wer, S1[0], S1[1], "wer")
    run_case("wer_100_x_12w", mojo.wer, upstream.wer, S2[0], S2[1], "wer")

    OUT["loadavg_at_end"] = os.getloadavg()
    with open(os.path.join(REPO, "bench", "analysis_raw.json"), "w") as f:
        json.dump(OUT, f, indent=1)
    print("wrote bench/analysis_raw.json")


if __name__ == "__main__":
    main()

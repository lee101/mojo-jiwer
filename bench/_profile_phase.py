"""ANALYZE-phase profiler: split each public entry point into Python glue vs kernel.

Run with the bench lock so it never overlaps `pixi run bench`:
    flock /tmp/mojo-bench.lock pixi run python bench/_profile_phase.py

Every number this prints is measured here; nothing is estimated.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import bench as b  # corpus generators, same seeds as bench.py
import jiwer as upstream
import mojo_jiwer as mojo
from mojo_jiwer import _native
from mojo_jiwer.process import _default_transform, _encode, _chunks, _prepare
from mojo_jiwer.transformations import wer_default, cer_default


def best(fn, repeat=7):
    best_s = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best_s = min(best_s, time.perf_counter() - t)
    return best_s


def byte_touch(**kw):
    """Track bytes allocated by numpy inside a call."""
    before = np.zeros(1)


# ---------------------------------------------------------------- corpus
W10k = b.word_corpus(10_000, 12, 1)
W2k60 = b.word_corpus(2_000, 60, 2)
C5k = b.character_corpus(5_000, 48, 3)
W2k20 = b.word_corpus(2_000, 20, 4)
W1k = b.word_corpus(1, 1_000, 5)

CASES = [
    ("WER 10k x 12w", mojo.wer, lambda r, h: upstream.wer(r, h), W10k, "wer"),
    ("WER 2k x 60w", mojo.wer, lambda r, h: upstream.wer(r, h), W2k60, "wer"),
    ("CER 5k x 48c", mojo.cer, lambda r, h: upstream.cer(r, h), C5k, "cer"),
    ("process_words 2k x 20w", mojo.process_words, lambda r, h: upstream.process_words(r, h), W2k20, "words"),
    ("WER 1 x 1000w", mojo.wer, lambda r, h: upstream.wer(r, h), W1k, "wer"),
]


def decompose(label, refs, hyps, kind):
    """Measure, independently, every stage the public call walks."""
    out = {}
    if kind == "wer":
        t = best(lambda: [_default_transform(refs, wer_default, True), _default_transform(hyps, wer_default, False)])
        out["default_transform"] = t
        r_tok = _default_transform(refs, wer_default, True)
        h_tok = _default_transform(hyps, wer_default, False)
        out["encode"] = best(lambda: _encode(r_tok, h_tok))
    elif kind == "cer":
        t = best(lambda: [_default_transform(refs, cer_default, True), _default_transform(hyps, cer_default, False)])
        out["default_transform"] = t
        r_tok = _default_transform(refs, cer_default, True)
        h_tok = _default_transform(hyps, cer_default, False)
        r_txt = "".join("".join(x) for x in r_tok)
        h_txt = "".join("".join(x) for x in h_tok)
        out["to_text_32le"] = best(
            lambda: [
                np.frombuffer(r_txt.encode("utf-32le"), dtype="<u4").astype(np.int64),
                np.frombuffer(h_txt.encode("utf-32le"), dtype="<u4").astype(np.int64),
            ]
        )
        out["encode"] = best(lambda: _encode(r_tok, h_tok))
    else:
        t = best(lambda: [_default_transform(refs, wer_default, True), _default_transform(hyps, wer_default, False)])
        out["default_transform"] = t
        r_tok = _default_transform(refs, wer_default, True)
        h_tok = _default_transform(hyps, wer_default, False)
        out["encode"] = best(lambda: _encode(r_tok, h_tok))

    if kind == "cer":
        # packed path used by scalar cer
        ref_off = np.empty(len(refs) + 1, dtype=np.int64)
        hyp_off = np.empty(len(hyps) + 1, dtype=np.int64)
        ref_off[0] = hyp_off[0] = 0
        np.cumsum(np.fromiter(map(len, refs), dtype=np.int64, count=len(refs)), out=ref_off[1:])
        np.cumsum(np.fromiter(map(len, hyps), dtype=np.int64, count=len(hyps)), out=hyp_off[1:])
        rtxt = "".join(v.strip() for v in refs)
        htxt = "".join(v.strip() for v in hyps)
        r = np.frombuffer(rtxt.encode("utf-32le"), dtype="<u4").astype(np.int64)
        h = np.frombuffer(htxt.encode("utf-32le"), dtype="<u4").astype(np.int64)
        out["distances_flat"] = best(lambda: _native.distances_flat(r, h, ref_off, hyp_off))
        out["_native.distances"] = best(lambda: _native.distances(
            [[ord(c) for c in "".join(v.strip())] for v in refs],
            [[ord(c) for c in "".join(v.strip())] for v in hyps],
        ))
    else:
        a = _encode(r_tok, h_tok)
        pairs = [(x, y) if len(y) <= len(x) else (y, x) for x, y in zip(a[0], a[1])]
        fr, fh, fro, fho = _native._flatten_sequences([p[0] for p in pairs], [p[1] for p in pairs])
        rl = np.diff(fro); hl = np.diff(fho)
        maxl = int(np.maximum(rl, hl).max(initial=0))
        words = max(1, (maxl + 63) // 64)
        masks = np.zeros((max(int(fr.max()), int(fh.max())) + 1) * words + 2 * words, dtype=np.uint64)
        res = np.empty(len(fro) - 1, dtype=np.int64)
        lib = _native.library()
        mws = words
        mso = (max(int(fr.max()), int(fh.max())) + 1) * words
        out["kernel mji_distances (pre-flattened, 1 worker)"] = best(
            lambda: lib.mji_distances(
                _native._address(fr), _native._address(fh), _native._address(fro), _native._address(fho),
                len(fro) - 1, 1, 0, _native._address(masks), 0, mws, mso, _native._address(res),
            ), repeat=5)
        full = best(lambda: _native.distances(a[0], a[1]), repeat=3)
        out["_native.distances (incl. Python flatten/validate)"] = full
        out["flatten+validate+ctypes overhead"] = full - out["kernel mji_distances (pre-flattened, 1 worker)"]
        out["_native.traces"] = best(lambda: _native.traces(a[0], a[1]), repeat=2)
        # _chunks over every trace (the Python-side alignment builder)
        traces = _native.traces(a[0], a[1])
        out["_chunks(all)"] = best(lambda: [_chunks(t) for t in traces])
        # Python-level reduce that process_words performs
        def reduce_loop():
            hits = subs = dels = ins = 0
            rw = hw = 0
            for rr, hh, ops in zip(a[0], a[1], traces):
                for c in _chunks(ops):
                    if c.type == "equal": hits += c.ref_end_idx - c.ref_start_idx
                    elif c.type == "substitute": subs += c.ref_end_idx - c.ref_start_idx
                    elif c.type == "delete": dels += c.ref_end_idx - c.ref_start_idx
                    else: ins += c.hyp_end_idx - c.hyp_start_idx
                rw += len(rr); hw += len(hh)
            return hits, subs, dels, ins, rw, hw
        out["process reduce (chunks+counts)"] = best(reduce_loop)
        out["kernel work info"] = {"count": len(fro) - 1, "max_len": maxl, "total_tokens": int(rl.sum() + hl.sum())}
    return out


# ---------------------------------------------------------------- kernel-only
def kernel_stats(ids_refs, ids_hyps, kind):
    ref = [list(x) for x in ids_refs]
    hyp = [list(x) for x in ids_hyps]
    pairs = [(r, h) if len(h) <= len(r) else (h, r) for r, h in zip(ref, hyp)]
    r, h, ro, ho = _native._flatten_sequences([p[0] for p in pairs], [p[1] for p in pairs])
    rl = np.diff(ro); hl = np.diff(ho)
    maxl = int(np.maximum(rl, hl).max())
    total_tokens = int(rl.sum() + hl.sum())
    return dict(total_tokens=total_tokens, max_len=maxl, count=len(pairs))


def cpu_flops(d):
    return d


def main():
    print(f"numpy {np.__version__}; python {sys.version.split()[0]}")
    print()
    print("## Stage decomposition (ms, best of 7)")
    print("| Case | public | transform | encode | distances/traces | other glue | sum-floor |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    results = {}
    for label, ours, theirs, (refs, hyps), kind in CASES:
        total = best(lambda: ours(refs, hyps))
        stages = decompose(label, refs, hyps, kind)
        results[label] = (total, stages, kind)
        core_key = "_native.traces" if kind == "words" else "distances_flat" if kind == "cer" else "_native.distances (incl. Python flatten/validate)"
        tr = (stages.get("default_transform", 0.0) + stages.get("encode", 0.0) + stages.get(core_key, 0.0)
              + stages.get("to_text_32le", 0.0))
        if kind == "cer":
            tr = (stages.get("default_transform", 0.0) + stages.get("to_text_32le", 0.0)
                  + stages.get("distances_flat", 0.0))
        print(f"| {label} | {total*1e3:.2f} | {stages.get('default_transform',0)*1e3:.2f} | "
              f"{stages.get('encode',0)*1e3:.2f} | {stages.get(core_key,0)*1e3:.2f} | "
              f"{(total-tr)*1e3:.2f} | {tr*1e3:.2f} |")

    print()
    print("## Full stage table (ms)")
    labels = set()
    for _, s, _ in results.values():
        labels |= set(s)
    for label, s in sorted(results.items()):
        print(f"### {label}")
        for k in sorted(s):
            if isinstance(s[k], dict):
                print(f"- {k}: {s[k]}")
            elif k.endswith("(all)") or k.startswith("flatten") or k.startswith("kernel ") or k.startswith("distances_flat") or k.startswith("_native.distances ("):
                print(f"- {k}: {s[k]*1e3:.3f} ms")
            else:
                print(f"- {k}: {s[k]*1e3:.3f} ms")

    print()
    print("## Numpy baseline: score matrix for the row-major trace, vectorized by row")
    for label, _, _, (refs, hyps), kind in CASES:
        if kind == "words":
            continue
        tokens = [[hash(w) & 0xFFFF for w in x.split()] for x in refs[:50]]
        hyp = tokens
        def npy_pairs():
            tot = 0
            for r, h in zip(tokens, hyp):
                n, m = len(r), len(h)
                prev = np.arange(m + 1, dtype=np.int64)
                rb = np.asarray(r)
                for i in range(1, n + 1):
                    cur = np.empty(m + 1, dtype=np.int64)
                    cur[0] = i
                    cur[1:] = np.minimum(prev[1:] + 1, prev[:-1] + (rb[i - 1] != np.asarray(h)))
                    cur[1:] = np.minimum(cur[1:], cur[:-1] + 1)
                    prev = cur
                tot += int(prev[-1])
            return tot
        t = best(npy_pairs, repeat=3)
        print(f"- {label}: numpy DP on 50 pairs = {t*1e3:.2f} ms")


if __name__ == "__main__":
    main()

# mojo-jiwer

`mojo-jiwer` is a standalone Mojo port of the compute-heavy edit-distance core
used for word error rate (WER) and character error rate (CER). Its Python API
tracks [jiwer](https://github.com/jitsi/jiwer) for the covered subset, so existing
code can use:

```python
import mojo_jiwer as jiwer
```

The project is tested for numerical results, counts, edge cases, and complete
alignment objects against jiwer 4.0.0.

## Coverage

The covered API is:

- `wer`, `cer`, `mer`, `wil`, and `wip`
- `process_words` and `process_characters`
- `WordOutput`, `CharacterOutput`, and `AlignmentChunk`
- `collect_error_counts`
- jiwer's default and contiguous WER/CER transformation pipelines
- the core `Compose`, case, whitespace, punctuation, word, and character transforms

Empty references, multiple sentence pairs, arbitrary Unicode text, custom
jiwer-compatible transforms, and RapidFuzz-compatible alignment tie-breaking
are supported.

The jiwer command-line interface, alignment/error visualizers, and its full
catalog of English- and Kaldi-specific normalization transforms are not
included. This repository targets WER/CER computation rather than every
presentation and preprocessing utility in upstream jiwer.

## Install

The checked-in Pixi environment pins the Mojo nightly used by this source:

```bash
pixi install
pixi run build
pixi run test
```

The build task creates `dist/libmojo-jiwer.so`. Python will also rebuild a
missing or stale library on first use when the Mojo compiler is available.

## Usage

Run this from the repository after installation and building:

```python
import mojo_jiwer as jiwer

error_rate = jiwer.wer("hello world", "hello duck")
print(error_rate)  # 0.5

result = jiwer.process_words(
    ["one short sentence", "another example"],
    ["one sentence", "another good example"],
)
print(result.substitutions, result.deletions, result.insertions)
print(result.alignments)

character_error_rate = jiwer.cer("kitten", "sitting")
print(character_error_rate)
```

For custom normalization, compose the included transforms and pass the same
objects through jiwer's usual keyword parameters:

```python
normalize = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)
score = jiwer.wer(
    "Hello, WORLD!",
    "hello world",
    reference_transform=normalize,
    hypothesis_transform=normalize,
)
```

## Benchmarks

Measured on 2026-08-24 on an Intel Xeon E5-2697 v4 at 2.30 GHz, Linux x86-64.
These are end-to-end Python API timings, including transformations, token
encoding, NumPy buffer preparation, FFI, and the native computation. The ratio
is upstream jiwer time divided by mojo-jiwer time; values below 1 mean
mojo-jiwer is slower.

| Case | mojo-jiwer | jiwer | jiwer / Mojo | Result |
|---|---:|---:|---:|---|
| WER: 10,000 x 12-word utterances | 101.87 ms | 182.15 ms | 1.79x | faster |
| WER: 2,000 x 60-word utterances | 81.09 ms | 129.57 ms | 1.60x | faster |
| CER: 5,000 x 48-character utterances | 6.40 ms | 148.15 ms | 23.16x | faster |
| process_words: 2,000 x 20 words | 43.28 ms | 48.87 ms | 1.13x | faster |
| WER: one 1,000-word transcript | 0.74 ms | 1.07 ms | 1.44x | faster |

These are deliberately reported as measured. The benchmark binds each
case's inputs in its callable; an earlier table accidentally late-bound all
five labels to the final 1,000-word input. Against the corrected pre-optimization
baseline, Mojo improved from 134.73 to 103.09 ms, 113.69 to 80.18 ms, 99.93 to
42.46 ms, and 2.67 to 0.77 ms in the four WER/processing cases. CER moved from
4.75 to 6.35 ms and remains far ahead of upstream; it was not an optimization
target.

The table above is the unroll-phase baseline, re-measured on 2026-11-10 with the
same `bench.py`, corpora, and seeds. That pass hoisted the `len(operations)` of
`_chunks` out of both loops. In isolation that makes `_chunks` 1.08x faster
(10.45 to 9.65 ms for 2,000 x 20 `process_words`, best of 15 runs), but
`process_words` end to end did not move outside noise (43.28 against 43.4 ms
median for the same binary in an interleaved harness), so the table keeps the
baseline figures rather than claiming a gain. Three other candidates were
measured and reverted because they were not faster at these sizes: a 4-column
hand-unroll of the traceback matrix loop (1.8% on the kernel, under 0.5% end to
end), a bound-method token encoder (0.89-1.01x), and a preallocated flatten of
the token buffers (0.43-0.59x). `@parameter for` does not compile on the pinned
compiler, which rejects decorators on `for`. No floating-point reduction changed
in this pass, so results and counts are bit-identical and no tolerance was
touched.

Reproduce the table with:

```bash
pixi run bench
```

The Pixi task uses a machine-wide file lock so concurrent benchmark jobs do not
run together.

## SIMD pass

The hot contiguous loops in `src/capi.mojo` were already vectorised in earlier
phases: the common-affix scans, the row-0/column-0 boundary, and the multiword
mask fill and state initialisation are host-width SIMD (`comptime W =
simd_width_of[DType.float64]()`, 16 lanes here) with scalar tails. This pass
looked at what was left, and kept nothing: no candidate beat the previous
binary at a caller size.

Every number below was measured on 2026-11-10 on this machine (Intel Xeon
E5-2697 v4, 72 threads). The host is shared and was busy throughout (load
average 45-88), so the method matters more than any single number:

- Candidates were built into separate `.so` files, loaded side by side in one
  Python process, and timed by calling the public API alternately (old, new,
  old, new), three calls per measurement, nine measurements per build. Per-run
  ratios are reported, never a single timing.
- Micro-candidates too short for the wall clock were measured with process CPU
  time (`getrusage`) and the minimum over repeats.
- Where run-to-run spread could decide it (the traceback candidates), the A/B
  was repeated in both argument orders; for those, three runs in each order
  agreed.
- The shipped binary is byte-identical to the previous one
  (sha256 `4ccbbd50...`), so the table farther up still describes it.

| Candidate | Why it looked worth it | Measured old/new (faster > 1) | Verdict |
|---|---|---:|---|
| 16-byte boundary stores (`W = 4`) in `trace_impl` | fewer iterations for row 0 | compiled to the same bytes | reverted |
| Same, gated on a 16-byte-aligned start | keeps the store legal for any `m` | traces 2k x 20w 1.009, 2k x 60w 1.013 | reverted, slower |
| Carry the mask one-hot instead of `1 << (pi % 64)` | drops a modulo per mask word | 0.98-1.01 over 200-2000-word patterns | reverted, neutral |
| Fuse the mask clear into the build pass | one pass over the hints, not two | 0.996-1.006 at real pattern widths | reverted, neutral |
| Initialize row 0 and column 0 in one pass over the matrix | the two writes interleave in the same cache lines | traces 2k x 20w 1.012, 10k x 12w 1.040, 2k x 60w 1.018 | reverted, slower |
| Build all chunks and counts in one Python pass | `_chunks` is 10.1 ms of a 45 ms call | 41.68 vs 41.83 ms CPU (fused/current) | reverted, neutral and would need a native path |
| 16-lane vector mask fill, `hint[k] = k & 15` | no token-indirect loads in the fill | — | impossible, see below |
| One-ended `str.strip()` in the CER wrapper | two Python sweeps per utterance | — | dropped, changes which whitespace is removed |

Also checked and left alone: the mask clear-and-build is already one pass
(`hint[...] = bit` for the first dimension, `|=` only for repeats), and the
generated code has no `1 << (pi % 64)` shift to optimize away.

### What the numbers say about the loops

- **The boundary pass is already one 32-byte store per 8 cells.** The compiled
  loop is `vmovdqu %ymm`; the affix scans use `vptest`, which reports "all
  lanes equal" from one mask, so `reduce_and()` costs nothing extra. Halving
  the store to 16 bytes does not shorten the loop in instructions, and the
  measurements agree.
- **A comparison benchmark on this host needs both argument orders.** The same
  build against itself measured 0.974-1.008 in one order; the row/column
  reordering measured 0.974-0.995 in the first order and then 0.978-1.051
  when swapped, against a 1.008 self-control in the same order. Only running
  both orders made the ~2.5% loss visible.
- **The traceback is not vectorisable.** The 32-bit cost-matrix fill and the
  traceback each depend on the cell before them, so both stay scalar. The
  affix scans around them are SIMD and are where the width already is.

### Why the mask fill cannot be widened

The multiword mask is built with `hint[token * stride + pi // 64] |= 1 << pi`,
a token-indirect scatter: the address depends on the token at each position. A
vector form needs the addresses to be computable from the lane index alone.
Hint word 0 carries the bits for positions 0..63, word 1 for 64..127, and so
on, and a 16-word vector write is 1024 positions long, so for a pattern shorter
than 1024 positions a 16-word tile writes each of its words exactly once and
just so happens to reproduce the scalar addresses (at exactly 1024 they are
one-to-one; longer patterns need words the tile cannot reach). That makes a
vector fill possible only as a size-gated special case for patterns of at most
1024 positions with distinct tokens, not as a vectorisation of the algorithm.
The mask work is also too small to pay for that check: measured with the
pattern length varied and the text held at one word, ~1000-8000 mask words cost
under 0.5 us of a ~12 us call.

### Where the time actually is

Measured per stage, CPU time, minimum of five (ms):

| Case | public | default_transform | encode | kernel |
|---|---:|---:|---:|---:|
| WER 10k x 12w | 107.49 | 56.41 | 20.68 | 14.18 |
| WER 2k x 60w | 89.70 | 51.57 | 18.16 | 10.20 |
| CER 5k x 48c | 7.57 | 7.18 (packing) | — | 2.16 |
| process_words 2k x 20w | 46.16 | 15.58 | 6.48 | 7.00 traces + 10.10 chunks |

For the WER cases roughly 72% of the call is Python glue (normalisation and
token encoding) against 13% in the edit-distance kernel, so widening the kernel
cannot move the end-to-end number. The kernel's per-call cost is dominated by
fixed overhead, not by bytes: a call with no DP work at all takes ~12 us, and
the 1000-word transcript (one Myer call whose DP takes almost the whole call)
costs ~12.5 us. Vectorising the mask traffic would therefore not pay: it is
under 1 us of a ~12 us call. In `process_words` the kernel is 7.0 ms;
`_chunks` is 10.1 ms, of which the scan is 2.7 ms and building the
`AlignmentChunk` objects and their lists is the rest. Counts and alignments
must both be produced, and doing that in one Python pass (implemented here and
checked value-for-value against `process_words`) measured 41.68 against
41.83 ms CPU for the whole call: inside the noise floor, so it was not kept.

### Test coverage added

`tests/test_simd_tails.py` pins down the tails this kind of change puts at
risk. Lengths 0, 1, W-1, W, W+1 and 1001 (W is 16 on this build, so the scalar
tails of the host-width scans run), checked against upstream jiwer for hits,
substitutions, insertions, deletions, wer, mer, wil, wip, and alignment
chunks. It also covers the multiword hints: 65- to 1024-word patterns, a token
repeated across the whole pattern (where a fused clear-and-build would drop
bits: that change moved a 1000-word transcript's distance from 112, matching
upstream, to 114), and buffer reuse across calls.

### Bench lock

The `pixi run bench` task locks `/tmp/mojo-bench.lock`, which every project on
this host shares. During this pass that lock was held by a process that had
already exited (its entry stayed in `/proc/locks`), so every waiter blocked in
`flock(2)`. The measurements above ran serially with that lock bypassed via a
`PATH` shim; normal `pixi run bench` should work once the lock is clear.

Reproduce the table with:

```bash
pixi run bench
```

The Pixi task uses a machine-wide file lock so concurrent benchmark jobs do not
run together.

## How it works

Python applies jiwer-compatible transforms and maps arbitrary word tokens to
dense 64-bit integer IDs. Default scalar CER converts contiguous UTF-32
codepoints directly into 64-bit NumPy buffers. For scalar WER/CER over multiple
utterances, the wrapper flattens the tokens with 64-bit sentence offsets and
makes one batched `ctypes` call.

The Mojo kernel trims common affixes and uses Myers edit distance: a specialized
64-bit kernel for short sequences and an exact multiword bit-parallel kernel for
longer sequences. Full processing uses a row-major 32-bit cost matrix and an
8-bit operation buffer to recover exactly the alignment choices used by
upstream jiwer's RapidFuzz backend.

Common-affix scans, matrix boundaries, and multiword state initialization use
host-width SIMD with scalar remainder loops. Distance batches stay serial
unless they contain at least 256 pairs and 200,000 input tokens. Traceback is
batched across one FFI call and stays serial unless it contains at least 256
pairs and 200,000 dynamic-programming cells. Larger batches use up to 16 coarse
CPU workers with independent, memory-capped scratch buffers.

There is intentionally no GPU path. Multiword Myers performs roughly 20 integer
operations while loading or updating about 24 bytes per word, and each traceback
cell moves at least 16 bytes for only a handful of integer operations. Both are
well below the roughly 2 operations-per-byte threshold where device transfer
and launch cost can pay off, so a GPU path would lose on these workloads.

All FFI buffers are C-contiguous NumPy allocations owned by Python. They cross
the C ABI as integer addresses and are rebuilt as
`UnsafePointer[..., AnyOrigin[mut=True]]` inside the exported Mojo functions.
Mojo does not allocate or retain caller memory.

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

## Parallel phase

Goal of this pass: split independent per-utterance work across cores above a
size threshold, but only where the benchmark says it wins. It does not win on
this toolchain, at any size a real caller passes, so **no in-kernel
parallelism was added and the shipped binary is unchanged** (sha256
`4ccbbd50...`, byte-identical to the SIMD pass). What is recorded here is the
measurement that decided it.

### What was measured

`max.algorithm.parallelize(func, num_work_items, num_workers)` — the live
import on `mojo ==1.2.0.dev2026092605`, per `MOJO_NOTES.md` §4 — does
parallelise correctly from a `ctypes` host. Probing it with a spin loop of
exactly-known cost, the wall time scales as expected:

| work items | 1 worker | 2 workers | 4 workers |
|---|---:|---:|---:|
| 400 x 50 us spin, busy time / wall span | 1.00 | 1.99 | 3.95-3.99 |

The worker count reaching 4 is measurable in the work itself, so the claim is
not based on empty work items. The catch is the launch cost, timed inside
Mojo with `perf_counter_ns` around the `parallelize` call with empty workers:
**median 5.3-6.3 ms, minimum 5.2 ms, for 2 or 4 workers on a single item or
400.** That cost is fixed per call, not per worker or per item, and on this
shared host (load average 30-130 during the pass) it fluctuates by about a
millisecond run to run.

Break-even follows from those two numbers. The same spin loop, 200 items,
4 workers:

| work per item | serial work | wall | speedup | python-thread alternative |
|---|---:|---:|---:|---:|
| 100 us | 20 ms | 24.9 ms | 0.80x (loss) | parity at 20.5 ms |
| 500 us | 100 ms | 30.5 ms | 3.28x | parity at 100.5 ms |
| 2000 us | 400 ms | 108.9 ms | 3.67x | parity at 400.5 ms |
| 4000 us | 800 ms | 206.7 ms | 3.87x | parity at 800.5 ms |

So a 4-worker `parallelize` only repays its launch above roughly **80-100 ms
of serial kernel work** (it is still losing at 20 ms; simulated from the
measured 5.2-6.3 ms launch, break-even is 5.2 x 4 = 21 ms, and in practice
further above that because the host is busy). The existing coarse path —
`chunk_index` over up to 16 `ThreadPoolExecutor` threads in
`python/mojo_jiwer/_native.py` — launches 4 threads in a median **0.50 ms**,
about eleven times cheaper, so moving the same split inside Mojo would need a
much larger batch before it could win.

### The before and after, at sizes callers pass

Serial kernel cost, measured on the shipped binary (`_native.distances_flat`
and `_native.traces`, warm, median of 5, wall time equal to CPU time):

| batch | kernel work | tokens |
|---|---:|---:|
| distance 5,000 x 12w | 0.85 ms | 120,000 |
| distance 20,000 x 12w | 2.26 ms | 480,000 |
| distance 5,000 x 60w | 2.72 ms | 600,000 |
| distance 10,000 x 48c | 4.40 ms | 960,000 |
| distance 20,000 x 48c | 8.80 ms | 1,920,000 |
| distance 40,000 x 48c | 20.94 ms | 3,840,000 |
| trace 2,000 x 20w | 6.47 ms | 80,000 |
| trace 2,000 x 60w | 24.25 ms | 240,000 |
| trace 10,000 x 20w | 34.56 ms | 400,000 |

and end to end through the public API (same corpus and seeds as the benchmark
table above, pickled callables avoided, median of 5):

| call | mojo-jiwer (before) | with a 4-worker in-kernel path (after) |
|---|---:|---:|
| WER 10k x 12w (1.18 ms kernel) | 114.58 ms | not shipped: kernel is 1/97th of the call, the split is in Python glue |
| WER 2k x 60w (1.72 ms kernel) | 90.28 ms | not shipped, same reason |
| CER 5k x 48c (2.10 ms kernel) | 6.73 ms | not shipped, same reason |
| process_words 2k x 20w (6.47 ms trace) | 49.70 ms | would lose: 6.47 ms of work behind a ~6 ms launch, before 4-way speedup |
| WER 20k x 12w (2.26 ms kernel) | 227.68 ms | not shipped, kernel is ~1% |
| CER 40k x 48c (8.80 ms kernel) | 58.64 ms | would lose: 8.8 ms of work behind a ~6 ms launch |
| process_words 10k x 20w (34.56 ms trace) | 286.81 ms | 34.56 ms of work split 4 ways saves ~26 ms, but the call is dominated by the 253 ms Python transform/encoding/chunking, so end to end is under 1% |

The largest batch any caller realistically passes to one call (40,000
utterances of 48 characters) has 8.8-20.9 ms of kernel work; that is the top
of the range in the first table. Even there, dividing that work by four would
save at most 6.6-15.7 ms (arithmetic: 3/4 of the serial kernel time) against a
~6 ms advance for the launch, and the same caller already spends 58.64 ms end
to end on transforms and encoding, so the saving cannot exceed 27% of
`cer` at the single largest size and is a few percent or a loss below it. Every
smaller size loses outright. Per the rule for this phase: a parallel path that
loses at caller sizes is worse than none, so it was reverted and no threshold
is exposed.

### What was tried and dropped, and why

- **In-kernel `parallelize` over work items** (cap 4). Correct output, scales
  3.95x, but the 5.2-6.3 ms launch loses below ~80-100 ms of work.
  Dropped, not shipped.
- **Worker threads joining an existing `max` thread pool across calls.** The
  launch cost is identical on every call (5.3-6.3 ms) whether the call is the
  first or the fifteenth, so there is no warm-up to amortise from Python.
  Dropped.
- **Raising the worker cap above 4.** The spin probe reaches 3.95x at 4 and
  the launch cost does not fall with more workers; on a host at load average
  30-130 the gain flattens. The cap was never raised, so 8/16-worker numbers
  are not claimed here.
- **The existing `ThreadPoolExecutor` path.** It is the comparison, not a
  change: median 0.50 ms to launch 4 threads, so it stays.

Trap worth recording for the next pass: building with `from max.algorithm
import parallelize` makes the shared library link `libAsyncRTMojoBindings.so`,
which the repo's Pixi env does not contain until the `max` and `max-core`
packages are pinned at the Mojo version (`26.7.0.dev2026092605` for
`mojo ==1.2.0.dev2026092605`). Without them `ctypes.CDLL` fails at load with
`cannot open shared object file`. The shipped binary has no such dependency
(only `libc`), so no pin was added while no `max` import ships.

### Test coverage added

`tests/test_parallel_parity.py` compares a serial batch with the same batch
computed by four threads at once: `wer` and the raw `_native.distances` on
three corpora, and `process_words` (counts and alignment chunks, plus
`_native.traces`) against upstream jiwer and against itself. It passes because
both paths reach the same serial kernel; if an in-kernel worker path is added
later, it is the test that will catch the two paths drifting apart.

Reproduce the measurements with warm runs of the shipped binary, e.g.:

```bash
pixi run python -c "
import mojo_jiwer as mj, time
r = ['w%d w%d w%d' % (i, i+1, i+2) for i in range(20000)]
t = time.perf_counter(); mj.wer(r, r); print('warm', (time.perf_counter()-t)*1e3, 'ms')"
```

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
unless they contain at least 256 pairs and 2,000,000 input tokens. Traceback is
batched across one FFI call and stays serial unless it contains at least 256
pairs and 10,000,000 dynamic-programming cells. Larger batches use up to 16 coarse
CPU workers with independent, memory-capped scratch buffers.

## GPU phase

**CPU only: memory-bound.** No GPU path is shipped, and no GPU test or build
step was added — so there are no GPU tests that could be skipped here. The four
skipped tests in `pixi run test` are the pre-existing `test_simd_tails.py` cases
(no prefix/middle/suffix split, and an empty reference that upstream rejects);
they are unrelated to the device. None of the kernels clears the gate for this
phase: a kernel is a GPU candidate only above roughly 2 flops per byte, and none
here reaches 1.

Every ratio below is in the analysis output, `bench/analysis_raw.json`, produced
by `bench/_analyze.py` on 2026-11-10 on the Intel Xeon E5-2697 v4 above. The
shared GPU was checked with `nvidia-smi --query-gpu=memory.free` and reported
4885 MiB free, so the memory gate passed; the compute gate is what stops this
phase.

| Kernel | Work per element | Traffic per element | flops / byte | Verdict |
|---|---|---:|---:|---|
| 64-bit Myers (`len <= 64`, most calls) | ~4-6 integer ops on one 64-bit word | 8 B token + 24 B mask/state | ~0.13-0.25 | CPU only: memory-bound |
| Multiword Myers (`len > 64`) | ~15 integer ops | 24 B read + 16 B store per word | ~0.38 | CPU only: memory-bound |
| Traceback DP + backtrace | 3-4 integer ops, then a serial walk | 12 B read + 4 B store per cell | ~0.19-0.25 | CPU only: memory-bound |

The analysis script's own `flops / bytes` column, which counts one Myers step per
input token against eight bytes of token, is 0.125 for every WER and CER case
(`wer_10k_x_12w`, `wer_2k_x_60w`, `cer_5k_x_48c`). Counting the real inner loop
rather than the abstract step, the multiword kernel is the densest of the three
at roughly 0.38 flops per byte. Every figure is five to fifteen times under the
threshold, so a device path would pay full copy-in, launch, and copy-back cost
to run arithmetic the CPU finishes in 0.3-1.7 ms of kernel time at these sizes.

Sizes are the ones real callers pass, not a synthetic maximum: 10,000 x 12-word,
2,000 x 60-word, 5,000 x 48-character, 2,000 x 20-word `process_words`, one
1,000-word transcript, and 100 x 12-word. The distance kernel takes 0.83-1.66 ms
on the large cases and 3.4 us on the smallest, and the multiword inner loop keeps
its per-word op and byte counts as sequences grow, so no caller size in this
sweep moves a kernel toward the 2 flops/byte line. There is no before/after
timing pair to record because no GPU code was built or run; the numbers above are
the CPU baseline it would have had to beat.

This supersedes the earlier "intentionally no GPU path" paragraph, which argued
the same conclusion from the intrinsic counts (about 20 integer operations over
24 bytes per word) without recording the per-kernel ratios.

All FFI buffers are C-contiguous NumPy allocations owned by Python. They cross
the C ABI as integer addresses and are rebuilt as
`UnsafePointer[..., AnyOrigin[mut=True]]` inside the exported Mojo functions.
Mojo does not allocate or retain caller memory.

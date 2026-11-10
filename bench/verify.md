# VERIFY pass — 2026-11-10

Machine: Intel Xeon E5-2697 v4 @ 2.30 GHz, Linux x86-64, 72 threads, shared
(load average 32–38 during these runs). Upstream `jiwer` is 4.0.0
(`importlib.metadata`), the version the README claims. Mojo
`1.2.0.dev2026092605`, Python 3.13.14 as pinned in `pixi.toml`.

All numbers below were measured in this session. Nothing is carried over from
earlier passes.

## Gate

```
pixi run build && pixi run test   -> 504 passed, 4 skipped   (before my test edit)
pixi run build && pixi run test   -> 568 passed, 4 skipped   (final, after)
```

The 4 skips are the pre-existing `tests/test_simd_tails.py` cases described in
the README (no prefix/middle/suffix split, empty reference upstream rejects).
There is no GPU test to skip: no GPU code exists.

## README benchmark table, re-measured

`pixi run bench`, run twice. Point estimates moved run to run (the host is
shared, and the 10k x 12w case swings 99–166 ms for mojo and 188–355 ms for
upstream), so the table below reports the two `bench.py` runs and an
interleaved best-of-9 / median-of-9 on each side (my harness, same corpora and
seeds as `bench/bench.py`).

| Case | README (mojo / jiwer) | bench run 1 | bench run 2 | interleaved best / median (mojo; jiwer) |
|---|---|---|---|---|
| WER: 10,000 x 12-word | 101.87 / 182.15 ms | 106.43 / 192.03 | 99.09 / 187.97 | 93.69 / 165.54 ; 187.36 / 355.33 |
| WER: 2,000 x 60-word | 81.09 / 129.57 ms | 83.12 / 137.57 | 93.93 / 132.92 | 72.95 / 75.77 ; 128.14 / 130.40 |
| CER: 5,000 x 48-character | 6.40 / 148.15 ms | 6.97 / 157.17 | 6.28 / 148.67 | 7.18 / 7.89 ; 148.66 / 152.54 |
| process_words: 2,000 x 20 | 43.28 / 48.87 ms | 44.21 / 51.80 | 42.12 / 48.90 | 40.91 / 42.47 ; 47.95 / 49.89 |
| WER: one 1,000-word | 0.74 / 1.07 ms | 0.76 / 1.08 | 0.75 / 1.06 | 0.74 / 0.78 ; 1.06 / 1.09 |

Every cell on both sides is inside ±20% of its README value, in every one of the
three measurements. Largest single deviation: jiwer on 10k x 12w, 192.03 ms
(+5.4%) in `bench.py`; the interleaved median (+95%) is a busy-host outlier, and
its best-of-9 is +2.9%. **No README table figure was corrected for being out of
tolerance**, because none was.

Direction of every row is unchanged: all five are still `faster`, and none was
slower than upstream in any run. No row was removed or reworded.

| Case | README ratio | run 1 | run 2 | interleaved best / median |
|---|---|---|---|---|
| WER 10k x 12w | 1.79x | 1.80x | 1.90x | 2.00x / 2.15x |
| WER 2k x 60w | 1.60x | 1.65x | 1.42x | 1.76x / 1.72x |
| CER 5k x 48c | 23.16x | 22.55x | 23.66x | 20.72x / 19.33x |
| process_words 2k x 20 | 1.13x | 1.17x | 1.16x | 1.17x / 1.17x |
| WER 1 x 1000w | 1.44x | 1.43x | 1.41x | 1.43x / 1.40x |

## Same input, same work — checked

- **Inputs.** `bench/bench.py` builds each corpus once and binds it as a
  default argument in both lambdas, so mojo and upstream receive the identical
  objects. Before timing, each case runs both sides once and asserts equality
  (`result.wer` for `process_words`, exact value otherwise); all five pass.
- **Both sides do the same algorithm.** Upstream `jiwer.wer`/`jiwer.cer` are
  thin wrappers around `process_words`/`process_characters` (checked with
  `inspect.getsource`). mojo's scalar `wer`/`cer` use a dedicated fast path
  (default transform without `Compose`, flattened UTF-32 buffers for CER, one
  batched FFI call). This is a real asymmetry in the *implementation* of the
  same public call, so I measured the matched-pair alternative:

  | Case | mojo wer/cer (shipped path) | mojo process_words | upstream wer/cer | upstream process_words |
  |---|---|---|---|---|
  | WER 10k x 12w | 92.03 ms | 160.19 | 302.77 | 202.45 |
  | WER 2k x 60w | 73.14 ms | 123.38 | 122.71 | 126.77 |
  | CER 5k x 48c | 6.01 ms | 40.51 | 146.83 | 49.57 |
  | WER 1 x 1000w | 0.72 ms | 2.74 | 1.02 | 1.03 |

  The table compares the same public call on both sides, and upstream's own
  `wer`/`cer` is not the slow side of a skipped step. Both sides are timed
  best-of-N through the public API; neither skips transformation, encoding,
  buffer preparation, FFI, or the native computation. Nothing is wrong with the
  comparison. The per-call choice of fast path is documented behavior of the
  module, and upstream has no equivalent shortcut to compare against.
- **Not measured on both sides:** nothing in the published table. The
  `bench/analysis_raw.json`, `.parwork`, `.simdwork`, `.probe` artifacts are
  scratch from earlier phases, not evidence, and were not used.

## Tests: parity with upstream, and history

`git log -p -- tests/` (five commits): every commit that touched `tests/` only
**added lines** — three new files (`test_degenerate_parity.py`,
`test_simd_tails.py`, `test_parallel_parity.py`) and two additive edits to
`test_parity.py` (+25 lines in `c62a10d`, +240 lines in `924444b`). No assertion
was removed, no test deleted, no tolerance raised, in any commit.

Gap found and closed: `test_word_measures_within_explicit_tolerance`,
`test_process_float_fields_within_tolerance`, and
`test_character_float_fields_within_tolerance` compared every float to upstream
only through `pytest.approx(..., abs=1e-12)`. A tolerance that generous can hide
a change in arithmetic order. I checked exact equality first — 0 differences
over the 16-string `SCALAR_GRID` x {wer, mer, wil, wip}, 0 over 200 random
duplicate-heavy corpora x 4 measures, 0 for the `process_words` float fields
and `cer` — then added exact `==` assertions in all three tests (one new
`test_word_measures_match_upstream_exactly`, plus exact checks inside the two
`process_*` tests). The original `approx` lines are kept, so the tests are now
strictly stronger: 504 -> 568 tests due to the extra parametrisation.

No production code was changed. `pixi run bench` output above was produced with
the shipped `dist/libmojo-jiwer.so` (sha256 `4ccbbd50…`, links only libc).

## README corrections

Only statements I could show to be wrong against the tree, no measured numbers
changed:

- Threshold paragraph (was "How it works", now unchanged in wording): said
  distance batching kicks in at "200,000 input tokens" and traceback at
  "200,000 dynamic-programming cells". `python/mojo_jiwer/_native.py:210` and
  `:273` require **2,000,000 tokens** and **10,000,000 cells**. Corrected to the
  code. Measured consequence: `process_words` 2,000 x 20w is 882,000 cells, so
  traceback stays serial there; the 2,000-pair trace batch measured 12.1 ms
  serial and its distance batch 4.5 ms.

| Thresholds stated as "200,000 tokens / 200,000 cells" | How it works | **Wrong in the README**, fixed: code requires 2,000,000 tokens (`_native.py:210`) and 10,000,000 cells (`_native.py:273`). Measured consequence: `process_words` 2,000 x 20w is 882,000 cells, so its traceback stays serial there. |

## What I could not reproduce, and did not put in the README

These are earlier-phase claims outside the benchmark table. Where I can, the
claim is contradicted; where I cannot, I say so. I left the historical sections
in place rather than rewriting them, but none of these should be treated as
verified by this pass.

| Claim | Where | Status this session |
|---|---|---|
| "the 1000-word transcript (one Myer call whose DP takes almost the whole call) costs ~12.5 us" | SIMD pass, §Where the time actually is | **Contradicted.** `_native.distances` for 1 x 1000w is ~349 us CPU; `_native.distance` ~778 us. The trace is 1000 operations, not a few. The sentence also says a no-DP call costs ~12 us, so "almost the whole call" cannot be ~12.5 us. |
| "WER 2k x 60w kernel 10.20 ms" and "kernel's per-call cost is dominated by fixed overhead" | same table | Measured 15.95 ms kernel this session (README stage-table kernel column reproduces for the other row: 14.13 ms vs 14.18 ms for 10k x 12w). Fixed overhead is not dominant at this size. |
| "WER 10k x 12w ... 1.18 ms kernel" | Parallel phase table | **Contradicted.** Measured 14.13 ms at those sizes. The related breakdown text ("kernel is 1/97th of the call") also fails: public 103.5 ms, prepare 86.2 ms, kernel 14.1 ms here, i.e. ~7%, and a large part of "prepare" is the token encoding that feeds the kernel. |
| "~72% of the call is Python glue" | SIMD pass | Node computed from stale numbers; my stage run gives public 103.5 / prepare 86.2 / kernel 14.1 ms, so the *upper bound* on glue is ~83% and the measured kernel share is ~14%. Did not establish a stable corrected split. |
| Parallel-phase end-to-end "before" times (114.58, 90.28, 6.73, 49.70, 227.68, 58.64, 286.81 ms) and kernel row times | Parallel phase | No A/B script for them is checked in, and the kernel rows disagree with the SIMD table and my measurements (e.g. trace 2k x 20w "6.47 ms" vs 11.4–12.1 ms measured). **Not reproduced.** |
| `max.algorithm.parallelize` 5.2–6.3 ms launch, 3.95x spin scaling, break-even tables | Parallel phase | Not re-run (would need the `max` packages, which are not pinned). The Python side is consistent with the conclusion: `ThreadPoolExecutor(4)` launch measured **2.01 ms median** for 4 threads here (README says 0.50 ms), still far below 5–6 ms. |
| SIMD candidate A/B ratios (1.009, 1.013, 0.98–1.01, …) | SIMD pass, candidate table | Not re-run; the measured candidates were built into untracked `.simdwork/*.so` scratch and the shipped binary is the baseline, so "reverted, slower/neutral" cannot be re-measured from the tree. |
| "host-width SIMD ... 16 lanes here" / `test_simd_tails.py` "W = 16 on this build" | How it works; simd tails docstring | Code uses `comptime W = simd_width_of[DType.float64]()`; on x86-64 without AVX-512 that resolves to 8 f64 lanes, not 16. **Not confirmed in this session** — nothing in the repo prints `W` — so this is flagged as unverified, not corrected. The tail test's lengths (0/1/15/16/17/1001) bracket both widths. |
| "shared GPU ... 4885 MiB free" | GPU phase | Measured **4056 MiB free** today. The compute-gate conclusion (all kernels under 2 flops/byte) is arithmetic and not time-dependent, so the verdict stands, but the memory figure is stale. |
| README "simd pass" says "a call with no DP work at all takes ~12 us" | SIMD pass | Not reproduced as stated. A non-empty batch with almost no DP (100 x 12w distances) measured 0.35 ms end to end through `_native.distances`, including buffer set-up; I did not construct a true zero-work call, so the 12 us figure is neither confirmed nor refuted. |

## Commands used

```
pixi run build && pixi run test
pixi run bench
pixi run python /tmp/verifyphase/interleaved.py 9     # interleaved re-measurement
pixi run python /tmp/verifyphase/stages.py           # per-stage CPU times
pixi run python /tmp/verifyphase/exactness.py        # exact-equality pre-check
git log -p -- tests/
```

Scratch scripts live under `/tmp/verifyphase/` and are not part of the repo.

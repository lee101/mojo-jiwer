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

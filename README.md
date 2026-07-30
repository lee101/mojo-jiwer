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

Measured on 2026-07-30 on an Intel Xeon E5-2697 v4 at 2.30 GHz, Linux x86-64.
These are end-to-end Python API timings, including transformations, token
encoding, NumPy buffer preparation, FFI, and the native computation. The ratio
is upstream jiwer time divided by mojo-jiwer time; values below 1 mean
mojo-jiwer is slower.

| Case | mojo-jiwer | jiwer | jiwer / Mojo | Result |
|---|---:|---:|---:|---|
| WER: 10,000 x 12-word utterances | 2.81 ms | 1.16 ms | 0.41x | slower |
| WER: 2,000 x 60-word utterances | 2.76 ms | 1.13 ms | 0.41x | slower |
| CER: 5,000 x 48-character utterances | 102.88 ms | 6.94 ms | 0.07x | slower |
| process_words: 2,000 x 20 words | 3.00 ms | 1.03 ms | 0.34x | slower |
| WER: one 1,000-word transcript | 2.78 ms | 1.04 ms | 0.37x | slower |

These are deliberately reported as measured: mature jiwer delegates to the
highly optimized RapidFuzz C++ implementation and is faster in every tested
case. Reproduce the table with:

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

The Mojo kernel trims common affixes, uses a 64-bit Myers edit-distance kernel
for short sequences, and falls back to two-row Wagner-Fischer dynamic
programming for longer sequences. Full processing uses a row-major 32-bit cost
matrix and an 8-bit operation buffer to recover exactly the alignment choices
used by upstream jiwer's RapidFuzz backend.

Common-affix scans and dynamic-programming row initialization use host-width
SIMD with scalar remainder loops. Distance batches stay serial unless they
contain at least 256 pairs and 200,000 input tokens; larger batches use up to
16 coarse CPU workers with independent scratch buffers.

There is intentionally no GPU path. Myers performs roughly a dozen integer
operations per 16 bytes loaded, while each dynamic-programming cell moves at
least 16 bytes for only a handful of integer operations. Both are well below
the roughly 2 operations-per-byte threshold where device transfer and launch
cost can pay off, so a GPU path would lose on these workloads.

All FFI buffers are C-contiguous NumPy allocations owned by Python. They cross
the C ABI as integer addresses and are rebuilt as
`UnsafePointer[..., AnyOrigin[mut=True]]` inside the exported Mojo functions.
Mojo does not allocate or retain caller memory.

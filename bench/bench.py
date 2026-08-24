"""Benchmark mojo-jiwer against upstream jiwer on identical inputs."""

from __future__ import annotations

import math
import os
import platform
import random
import sys
import time

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

import jiwer as upstream  # noqa: E402
import mojo_jiwer as mojo  # noqa: E402


def best_time(function, repeat: int = 5) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def word_corpus(count: int, length: int, seed: int):
    rng = random.Random(seed)
    vocabulary = [f"word{i}" for i in range(256)]
    references = []
    hypotheses = []
    for _ in range(count):
        reference = rng.choices(vocabulary, k=length)
        hypothesis = reference.copy()
        for index in range(0, length, 9):
            hypothesis[index] = vocabulary[rng.randrange(len(vocabulary))]
        references.append(" ".join(reference))
        hypotheses.append(" ".join(hypothesis))
    return references, hypotheses


def character_corpus(count: int, length: int, seed: int):
    rng = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyz "
    references = []
    hypotheses = []
    for _ in range(count):
        reference = [rng.choice(alphabet) for _ in range(length)]
        hypothesis = reference.copy()
        for index in range(0, length, 11):
            hypothesis[index] = rng.choice(alphabet)
        references.append("".join(reference))
        hypotheses.append("".join(hypothesis))
    return references, hypotheses


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as file:
            for line in file:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main() -> None:
    cases = []

    reference, hypothesis = word_corpus(10_000, 12, 1)
    cases.append(
        (
            "WER: 10,000 x 12-word utterances",
            lambda reference=reference, hypothesis=hypothesis: mojo.wer(reference, hypothesis),
            lambda reference=reference, hypothesis=hypothesis: upstream.wer(reference, hypothesis),
        )
    )

    reference, hypothesis = word_corpus(2_000, 60, 2)
    cases.append(
        (
            "WER: 2,000 x 60-word utterances",
            lambda reference=reference, hypothesis=hypothesis: mojo.wer(reference, hypothesis),
            lambda reference=reference, hypothesis=hypothesis: upstream.wer(reference, hypothesis),
        )
    )

    reference, hypothesis = character_corpus(5_000, 48, 3)
    cases.append(
        (
            "CER: 5,000 x 48-character utterances",
            lambda reference=reference, hypothesis=hypothesis: mojo.cer(reference, hypothesis),
            lambda reference=reference, hypothesis=hypothesis: upstream.cer(reference, hypothesis),
        )
    )

    reference, hypothesis = word_corpus(2_000, 20, 4)
    cases.append(
        (
            "process_words: 2,000 x 20 words",
            lambda reference=reference, hypothesis=hypothesis: mojo.process_words(reference, hypothesis),
            lambda reference=reference, hypothesis=hypothesis: upstream.process_words(reference, hypothesis),
        )
    )

    reference, hypothesis = word_corpus(1, 1_000, 5)
    cases.append(
        (
            "WER: one 1,000-word transcript",
            lambda reference=reference, hypothesis=hypothesis: mojo.wer(reference, hypothesis),
            lambda reference=reference, hypothesis=hypothesis: upstream.wer(reference, hypothesis),
        )
    )

    print(f"Machine: {cpu_name()}; {platform.system()} {platform.machine()}")
    print()
    print("| Case | mojo-jiwer | jiwer | jiwer / Mojo | Result |")
    print("|---|---:|---:|---:|---|")
    for name, ours, theirs in cases:
        ours_result = ours()
        theirs_result = theirs()
        if hasattr(ours_result, "wer"):
            assert ours_result.wer == theirs_result.wer
        else:
            assert ours_result == theirs_result
        mojo_seconds = best_time(ours)
        upstream_seconds = best_time(theirs)
        ratio = upstream_seconds / mojo_seconds
        result = "faster" if ratio > 1 else "slower"
        print(
            f"| {name} | {mojo_seconds * 1e3:.2f} ms | "
            f"{upstream_seconds * 1e3:.2f} ms | {ratio:.2f}x | {result} |"
        )


if __name__ == "__main__":
    main()

"""End-to-end public API timings for the benchmark corpus (shipped binary)."""

import os
import statistics
import sys
import time

sys.path.insert(0, "/nvme0n1-disk/code/mojo-jiwer/python")
sys.path.insert(0, "/nvme0n1-disk/code/mojo-jiwer/bench")
os.environ["MOJO_JIWER_LIB"] = "/nvme0n1-disk/code/mojo-jiwer/dist/libmojo-jiwer.so"
import mojo_jiwer as mojo  # noqa: E402
from bench import word_corpus, character_corpus  # noqa: E402


def measure(fn, repeat=5):
    wall, cpu = [], []
    for _ in range(repeat):
        w = time.perf_counter()
        c = time.process_time()
        fn()
        wall.append(time.perf_counter() - w)
        cpu.append(time.process_time() - c)
    return statistics.median(wall), statistics.median(cpu)


cases = []
r, h = word_corpus(10_000, 12, 1)
cases.append(("WER 10k x 12w", lambda r=r, h=h: mojo.wer(r, h)))
r, h = word_corpus(2_000, 60, 2)
cases.append(("WER 2k x 60w", lambda r=r, h=h: mojo.wer(r, h)))
r, h = character_corpus(5_000, 48, 3)
cases.append(("CER 5k x 48c", lambda r=r, h=h: mojo.cer(r, h)))
r, h = word_corpus(2_000, 20, 4)
cases.append(("process_words 2k x 20w", lambda r=r, h=h: mojo.process_words(r, h)))
r, h = word_corpus(20_000, 12, 11)
cases.append(("WER 20k x 12w", lambda r=r, h=h: mojo.wer(r, h)))
r, h = character_corpus(40_000, 48, 13)
cases.append(("CER 40k x 48c", lambda r=r, h=h: mojo.cer(r, h)))
r, h = word_corpus(10_000, 20, 17)
cases.append(("process_words 10k x 20w", lambda r=r, h=h: mojo.process_words(r, h)))

print("| call | wall (ms) | cpu (ms) |")
print("|---|---:|---:|")
for name, fn in cases:
    w, c = measure(fn)
    print(f"| {name} | {w*1e3:.2f} | {c*1e3:.2f} |", flush=True)

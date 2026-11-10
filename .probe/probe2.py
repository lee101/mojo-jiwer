import sys
from dataclasses import asdict

import numpy as np

sys.path.insert(0, "/nvme0n1-disk/code/mojo-jiwer/python")
import jiwer as up
import mojo_jiwer as mj


def outcome(fn, *args):
    try:
        value = fn(*args)
        return ("ok", value)
    except Exception as exc:  # noqa: BLE001
        return ("err", type(exc).__name__, str(exc))


def show(label, a, b):
    if a[0] != b[0] or (a[0] == "err" and a[1] != b[1]):
        print(f"DIVERGE {label}")
        print(f"   mojo: {str(a)[:250]}")
        print(f"   up  : {str(b)[:250]}")


for bad in [
    lambda x: [],
    lambda x: {},
    lambda x: "",
    lambda x: [["a"], ["b"]],
    lambda x: [[1]],
    lambda x: "abc",
]:
    a = outcome(mj.wer, "a b", "a b", bad, bad)
    b = outcome(up.wer, "a b", "a b", bad, bad)
    show(f"transform->{bad('x')!r}", a, b)

for ref, hyp in [
    (float("nan"), "a"),
    ("a", float("inf")),
    ([float("nan")], ["a"]),
    (None, ""),
    ("", None),
    (0, 0),
    (["a", 1], ["a", "b"]),
    ([[1.5]], [["a"]]),
]:
    for name in ["wer", "cer", "process_words", "process_characters"]:
        a = outcome(getattr(mj, name), ref, hyp)
        b = outcome(getattr(up, name), ref, hyp)
        show(f"{name} {ref!r} {hyp!r}", a, b)

for proc in ["process_words", "process_characters"]:
    a = outcome(lambda: tuple(type(d).__name__ for d in mj.collect_error_counts(getattr(mj, proc)(["abc"], ["abd"]))))
    b = outcome(lambda: tuple(type(d).__name__ for d in up.collect_error_counts(getattr(up, proc)(["abc"], ["abd"]))))
    print(proc, "collect types mojo:", a, "up:", b)
    a = outcome(lambda: [dict(d) for d in mj.collect_error_counts(getattr(mj, proc)(["the fast fox"], ["the slow red fox"]))])
    b = outcome(lambda: [dict(d) for d in up.collect_error_counts(getattr(up, proc)(["the fast fox"], ["the slow red fox"]))])
    show(f"collect {proc}", a, b)

from mojo_jiwer import AlignmentChunk as A1
from jiwer import AlignmentChunk as A2
for args in [
    ("replace", 0, 1, 0, 1),
    ("unknown", 0, 1, 0, 1),
    ("equal", 2, 1, 0, 1),
    ("equal", 0, 1, 2, 1),
    (None, 0, 0, 0, 0),
    ("equal", 0, 0, 0, 0),
    ("insert", 0, 0, 0, 3),
]:
    a = outcome(lambda: asdict(A1(*args)))
    b = outcome(lambda: asdict(A2(*args)))
    show(f"AlignmentChunk{args}", a, b)

import random
rng = random.Random(11)
bad = 0
for trial in range(600):
    k = rng.choice([0, 1, 2, 3, 10, 40, 70])
    alphabet = "abc" if trial % 3 == 0 else "abcdefg"
    r = " ".join(rng.choices(alphabet, k=k))
    h = " ".join(rng.choices(alphabet, k=k))
    a = outcome(mj.process_words, r, h)
    b = outcome(up.process_words, r, h)
    if a[0] == "ok" and b[0] == "ok":
        if asdict(a[1]) != asdict(b[1]):
            bad += 1
            print(f"DIVERGE process_words {r!r} / {h!r}")
            for key in asdict(a[1]):
                if asdict(a[1])[key] != asdict(b[1])[key]:
                    print("  key", key, asdict(a[1])[key], asdict(b[1])[key])
    else:
        show(f"process_words {r!r} {h!r}", a, b)
print("random word divergences:", bad)

for n, m in [(63, 64), (64, 64), (127, 130), (200, 260), (300, 300)]:
    r = " ".join(str(i % 5) for i in range(n))
    h = " ".join(str((i * 7) % 5) for i in range(m))
    a = outcome(mj.process_words, r, h)
    b = outcome(up.process_words, r, h)
    if a[0] == "ok" and b[0] == "ok":
        if asdict(a[1]) != asdict(b[1]):
            print(f"DIVERGE long process_words {n}x{m}")
            for key in asdict(a[1]):
                if asdict(a[1])[key] != asdict(b[1])[key]:
                    print("  key", key, str(asdict(a[1])[key])[:300], "|", str(asdict(b[1])[key])[:300])
    else:
        show(f"long {n}x{m}", a, b)
print("done")

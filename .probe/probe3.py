import math
import random
import sys
from dataclasses import asdict

import numpy as np

sys.path.insert(0, "/nvme0n1-disk/code/mojo-jiwer/python")
import jiwer as up
import mojo_jiwer as mj


def outcome(fn, *args):
    try:
        return ("ok", fn(*args))
    except Exception as exc:  # noqa: BLE001
        return ("err", type(exc).__name__, str(exc))


def cmp(label, a, b):
    if a[0] != b[0] or (a[0] == "err" and a[1] != b[1]):
        print(f"DIVERGE {label}: mojo={str(a)[:180]} up={str(b)[:180]}")


# 1. error message text + class for mismatched sentence counts
for refs, hyps in [
    (["one", "two"], ["one"]),
    ([], ["one"]),
    (["one"], []),
    (["one", "two", "three"], ["one"]),
]:
    a = outcome(mj.wer, refs, hyps)
    b = outcome(up.wer, refs, hyps)
    print("count mismatch", refs, hyps)
    print("  mojo", a)
    print("  up  ", b)

# 2. AlignmentChunk error text
from mojo_jiwer import AlignmentChunk as A1
from jiwer import AlignmentChunk as A2
for args in [("unknown", 0, 1, 0, 1), ("equal", 2, 1, 0, 1), ("equal", 0, 1, 2, 1)]:
    a = outcome(lambda: asdict(A1(*args)))
    b = outcome(lambda: asdict(A2(*args)))
    print("chunk", args, "| mojo", a, "| up", b)

# 3. empty input shapes: full asdict parity
cases = [
    ("", ""),
    ([], []),
    ([""], [""]),
    ([], [""]),
    ([""], []),
    (["a", "", "b"], ["a", "b"]),
    (None, None),
]
for ref, hyp in cases:
    a = outcome(mj.process_words, ref, hyp)
    b = outcome(up.process_words, ref, hyp)
    ok = a[0] == b[0] and (a[0] == "err" or asdict(a[1]) == asdict(b[1]))
    print("empty-shape", repr(ref), repr(hyp), "match" if ok else f"MISMATCH {a} | {b}")

# 3b. exact return values for degenerate scalar input
for name in ["wer", "cer", "mer", "wil", "wip"]:
    for ref, hyp in [("", ""), ([], []), ("", "hello"), ("hello", ""), ([], ["hello"]), (["hello"], [])]:
        a = outcome(getattr(mj, name), ref, hyp)
        b = outcome(getattr(up, name), ref, hyp)
        print(name, repr(ref), repr(hyp), "mojo", a, "up", b)

# 4. non-mapping: NaN/inf typed input
for ref, hyp in [(float("nan"), "a"), ("a", float("inf")), ([float("nan")], ["a"]), (np.float64("nan"), "a")]:
    for name in ["wer", "cer"]:
        a = outcome(getattr(mj, name), ref, hyp)
        b = outcome(getattr(up, name), ref, hyp)
        print("nan/inf", name, repr(ref), repr(hyp), "mojo", a, "up", b)

# 5. unicode / combining / astral
for ref, hyp in [("e\u0301", "é"), ("e\u0301", "e"), ("😀", ""), ("", "😀"), ("ß", "ss"), ("İ", "i"), ("\x00", "")]:
    a = outcome(mj.process_characters, ref, hyp)
    b = outcome(up.process_characters, ref, hyp)
    ok = a[0] == b[0] and (a[0] == "err" or asdict(a[1]) == asdict(b[1]))
    if not ok:
        print("unicode MISMATCH", repr(ref), repr(hyp), a, b)

# 6. heavy duplicate-token alignment parity (tie-breaking)
rng = random.Random(5)
for trial in range(200):
    alphabet = rng.choice(["a", "a b", "x"])
    r = " ".join(rng.choices(alphabet.split(), k=rng.randrange(0, 9)))
    h = " ".join(rng.choices(alphabet.split(), k=rng.randrange(0, 9)))
    a = outcome(mj.process_words, r, h)
    b = outcome(up.process_words, r, h)
    if not (a[0] == b[0] and asdict(a[1]) == asdict(b[1])):
        print("dup MISMATCH", repr(r), repr(h))
        for key in asdict(a[1]):
            if asdict(a[1])[key] != asdict(b[1])[key]:
                print("   ", key, asdict(a[1])[key], "|", asdict(b[1])[key])
print("probe3 done")

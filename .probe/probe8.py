import sys

sys.path.insert(0, "/nvme0n1-disk/code/mojo-jiwer/python")
import jiwer as up
import mojo_jiwer as mj
import numpy as np

from mojo_jiwer import _native


def outcome(fn):
    try:
        return ("ok", fn())
    except Exception as exc:  # noqa: BLE001
        return ("err", type(exc).__name__, str(exc)[:90])


# native kernel with absurd token ids and negative ids
print("native max token", outcome(lambda: _native.distance([2**62], [2**62 - 1])))
print("native negative", outcome(lambda: _native.distance([-1], [1])))
print("native empty", outcome(lambda: _native.distance([], [])))
print("native huge ids both sides", outcome(lambda: _native.distance([10**6, 2], [10**6, 3])))
print("native trace huge", outcome(lambda: _native.trace([10**6, 2], [10**6, 3])))

# float tokens through the public API using a transform that emits floats -> both should reject
def floaty(x):
    return [[1.5]]

print("float transform mojo", outcome(lambda: mj.wer("a", "a", floaty, floaty)), "up", outcome(lambda: up.wer("a", "a", floaty, floaty)))

# NaN as a *word* string
print("nan word", outcome(lambda: mj.wer("nan inf", "nan")), outcome(lambda: up.wer("nan inf", "nan")))

# nan/inf as float ids passed directly to _native.distances
print("native float ids", outcome(lambda: _native.distances([[float("nan")]], [[1.0]])))

# unicode normalisation forms & whitespace-only
for ref, hyp in [(" ", " "), ("\u00a0", " "), ("a\u00a0b", "a b"), ("\t", " "), ("\r\n", "")]:
    a = outcome(lambda: (mj.wer(ref, hyp), mj.cer(ref, hyp)))
    b = outcome(lambda: (up.wer(ref, hyp), up.cer(ref, hyp)))
    print(repr(ref), repr(hyp), a, b, "match" if a == b else "MISMATCH")

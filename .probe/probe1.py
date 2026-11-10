import sys
from dataclasses import asdict

sys.path.insert(0, "/nvme0n1-disk/code/mojo-jiwer/python")
import jiwer as up
import mojo_jiwer as mj


def outcome(fn, *args):
    try:
        value = fn(*args)
        if hasattr(value, "__dataclass_fields__"):
            value = asdict(value)
        return ("ok", value)
    except Exception as exc:  # noqa: BLE001
        return ("err", type(exc).__name__, str(exc))


def same(a, b):
    if a[0] != b[0]:
        return False
    if a[0] == "err":
        return a[1] == b[1]
    return a[1] == b[1]


cases = [
    (None, None),
    ("", ""),
    ([], []),
    ([""], [""]),
    ([], [""]),
    ([""], []),
    ("a", "a"),
    ("a", ""),
    ("", "a"),
    (" a ", " a "),
    ("\t\n", ""),
    (["a b"], ["a b", "c"]),
    ("hello", "hello"),
    (["a", "", "b"], ["a", "b"]),
    (["one", "two"], ["one"]),
    ("😀😀", "😀"),
    ("e\u0301", "é"),
    (["hi\tthere"], ["hi there"]),
    (["a  b"], ["a b"]),
    (123, "a"),
    ([1], ["a"]),
    (b"abc", b"abc"),
    (["abc"], __import__("numpy").array(["abc"])),
    (["a b c"] * 300, ["a b d"] * 300),
    ("x" * 300, "y" * 300),
]

names = ["wer", "cer", "mer", "wil", "wip", "process_words", "process_characters"]
for name in names:
    for ref, hyp in cases:
        a = outcome(getattr(mj, name), ref, hyp)
        b = outcome(getattr(up, name), ref, hyp)
        if not same(a, b):
            print(f"DIVERGE {name} ref={ref!r:.60} hyp={hyp!r:.60}")
            print(f"   mojo: {str(a)[:300]}")
            print(f"   up  : {str(b)[:300]}")

tr_cases = [None, 3, 3.5, ["a", None], [], "a"]
for name in [
    "RemoveMultipleSpaces",
    "Strip",
    "ToLowerCase",
    "ToUpperCase",
    "RemovePunctuation",
    "ReduceToListOfListOfWords",
    "ReduceToListOfListOfChars",
    "ReduceToSingleSentence",
]:
    ours = getattr(mj, name)()
    theirs = getattr(up, name)()
    for arg in tr_cases:
        a = outcome(ours, arg)
        b = outcome(theirs, arg)
        if not same(a, b):
            print(f"DIVERGE transform {name}({arg!r})")
            print(f"   mojo: {str(a)[:200]}")
            print(f"   up  : {str(b)[:200]}")
print("probe1 done")

import sys

sys.path.insert(0, "/nvme0n1-disk/code/mojo-jiwer/python")
import jiwer as up
import mojo_jiwer as mj


class Raising:
    def __init__(self, value):
        self.value = value

    def __call__(self, sentence):
        return self.value


print("=== invalid transform output ===")
for value in [[], {}, "", [["a"], ["b"]], [[1]], "abc", [1, 2], [[None]]]:
    for name, module in [("mojo", mj), ("up", up)]:
        try:
            result = module.wer("a b", "a b", Raising(value), Raising(value))
            print(repr(value), name, "->", result)
        except Exception as exc:  # noqa: BLE001
            print(repr(value), name, "-> raises", type(exc).__name__, str(exc)[:70])

print("=== collect_error_counts defaults ===")
for proc in ["process_words", "process_characters"]:
    ours = mj.collect_error_counts(getattr(mj, proc)("", ""))
    theirs = up.collect_error_counts(getattr(up, proc)("", ""))
    print(proc, [type(x).__name__ for x in ours], [x.default_factory for x in ours])
    print(proc, [type(x).__name__ for x in theirs], [x.default_factory for x in theirs])

print("=== delimiter ===")
for name in ["ReduceToListOfListOfWords", "ReduceToSingleSentence"]:
    for d in ["", " "]:
        for ref in [["a b"], ["a", "b"]]:
            try:
                a = getattr(mj, name)(d)(ref)
            except Exception as exc:  # noqa: BLE001
                a = f"raises {type(exc).__name__} {exc}"
            try:
                b = getattr(up, name)(d)(ref)
            except Exception as exc:  # noqa: BLE001
                b = f"raises {type(exc).__name__} {exc}"
            print(name, repr(d), ref, "| mojo", a, "| up", b)

print("=== [None] and [1] per transform ===")
for name in ["RemoveMultipleSpaces", "Strip", "ToLowerCase", "ToUpperCase", "RemovePunctuation",
             "ReduceToListOfListOfWords", "ReduceToListOfListOfChars", "ReduceToSingleSentence"]:
    for arg in [[None], [1]]:
        for label, module in [("mojo", mj), ("up", up)]:
            try:
                print(name, arg, label, "->", getattr(module, name)()(arg))
            except Exception as exc:  # noqa: BLE001
                print(name, arg, label, "-> raises", type(exc).__name__)

print("=== ReduceToSingleSentence none ===")
for label, module in [("mojo", mj), ("up", up)]:
    try:
        print(label, module.ReduceToSingleSentence()(["a", None]))
    except Exception as exc:  # noqa: BLE001
        print(label, "raises", type(exc).__name__, exc)

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
        return ("err", type(exc).__name__, str(exc)[:90])


# scalar / buffer inputs that are neither str nor list
for ref, hyp in [(3, "ab"), ("ab", 3), (3, 3), (b"ab", b"ac"), (b"ab", "ac"), (1.5, 2.5), (("a",), ("a", 2))]:
    print("pair", repr(ref), repr(hyp))
    for name in ["wer", "cer", "mer", "wil", "wip"]:
        print("   ", name, "mojo", outcome(getattr(mj, name), ref, hyp), "| up", outcome(getattr(up, name), ref, hyp))

# degenerate lists inside an otherwise valid list
for ref, hyp in [([""], ["a"]), ([""], [""]), ([], [""]), ([" ", " "], [" "]), (["a", ""], ["a", "a"])]:
    print("list", ref, hyp, "| cer mojo", outcome(mj.cer, ref, hyp), "up", outcome(up.cer, ref, hyp))

# process_characters on empty lists -> alignment structure
print("pc []", asdict(mj.process_characters([], [])), asdict(up.process_characters([], [])))
print("pc nameless nan char", outcome(mj.process_characters, "\ud800", "\ud800"), outcome(up.process_characters, "\ud800", "\ud800"))

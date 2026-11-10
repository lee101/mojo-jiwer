import sys

sys.path.insert(0, "/nvme0n1-disk/code/mojo-jiwer/python")
import jiwer as up
import mojo_jiwer as mj

for proc in ["process_words", "process_characters"]:
    ours = mj.collect_error_counts(getattr(mj, proc)("a b", "a c"))
    theirs = up.collect_error_counts(getattr(up, proc)("a b", "a c"))
    for label, bag in (("mojo", ours), ("up", theirs)):
        subs = bag[0]
        print(label, "missing-key lookup:", subs[("zz", "yy")], "| factory int?")
        # prove the observable difference: int() yields the same value, but the
        # factory identity differs; upstream's is not `int`.
    print(proc, "factory is int:", [bag.default_factory is int for bag in ours],
          [bag.default_factory is int for bag in theirs])
    print(proc, "defaults:", [bag.default_factory() for bag in ours],
          [bag.default_factory() for bag in theirs])

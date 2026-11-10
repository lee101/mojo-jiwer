import sys

sys.path.insert(0, "/nvme0n1-disk/code/mojo-jiwer/python")
import jiwer as up
import mojo_jiwer as mj


def outcome(fn):
    try:
        return ("ok", fn())
    except Exception as exc:  # noqa: BLE001
        return ("err", type(exc).__name__)


for depth in [10, 100, 400, 900, 2000, 20000]:
    node = "a"
    for _ in range(depth):
        node = [node]

    def transform(value, node=node):
        return node

    print(depth, "mojo", outcome(lambda: mj.wer("a", "a", transform, transform)), "up", outcome(lambda: up.wer("a", "a", transform, transform)))

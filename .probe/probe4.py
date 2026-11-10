import os
import sys
from pathlib import Path

site = Path("/nvme0n1-disk/code/mojo-jiwer/.pixi/envs/default/lib/python3.13/site-packages")
sys.path.insert(0, str(site))
import jiwer as up

count = 0
for path in site.glob("*"):
    if path.name in {"jiwer", "rapidfuzz"} or path.name.startswith(("jiwer-", "rapidfuzz-")):
        for file in path.rglob("*"):
            if file.is_file() and str(site / "jiwer") not in str(file):
                print(file)
print("---- test module presence ----")
for name in ["test_measures", "test_process", "test_transforms", "tests"]:
    try:
        __import__(f"jiwer.{name}")
        print("importable", name)
    except Exception as exc:
        print("no", name, type(exc).__name__)
print("recursion limit:", sys.getrecursionlimit())
print("depth of nesting accepted:")
for depth in [100, 400, 1000, 3000, 10000, 20000, 100000]:
    node = "a"
    for _ in range(depth):
        node = [node]
    try:
        up._is_list_of_list_of_strings(node) if hasattr(up, "_is_list_of_list_of_strings") else None
        from jiwer.process import _is_list_of_list_of_strings
        _is_list_of_list_of_strings(node)
        print(depth, "ok")
    except RecursionError:
        print(depth, "RecursionError")

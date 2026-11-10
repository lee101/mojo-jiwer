import ctypes

import numpy as np

lib = ctypes.CDLL("/nvme0n1-disk/code/mojo-jiwer/.parwork/counter.so")
lib.counter.argtypes = [ctypes.c_int64] * 4
lib.counter.restype = ctypes.c_int64

for n in (4, 5, 8, 16, 100, 1000, 10000):
    for workers in (1, 2, 3, 4, 8):
        counts = np.zeros(max(n, 1), dtype=np.int64)
        lib.counter(counts.ctypes.data, n, workers, 0)
        distinct = int((counts > 0).sum())
        total = int(counts.sum())
        print(f"n={n:6d} workers={workers}: items hit={distinct:6d}/{n:<6d} total={total:6d}"
              f" {'OK' if distinct == n and total == n else 'BAD'}", flush=True)

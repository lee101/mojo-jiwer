import ctypes

import numpy as np

lib = ctypes.CDLL("/nvme0n1-disk/code/mojo-jiwer/.parwork/reduce.so")
lib.par_update.argtypes = [ctypes.c_int64] * 3
lib.par_update.restype = ctypes.c_int64
for n in (4, 8, 100, 1000, 10000, 100000):
    for workers in (1, 2, 4):
        v = np.arange(1, n + 1, dtype=np.int64)
        expected = np.where(v % 4 == 0, -v, v)
        total = lib.par_update(v.ctypes.data, n, workers)
        ok = bool((v == expected).all()) and total == int(expected.sum())
        print(f"n={n:6d} workers={workers}: {'OK' if ok else 'BAD'} total={total}",
              flush=True)

import ctypes

import numpy as np

k2 = ctypes.CDLL("/nvme0n1-disk/code/mojo-jiwer/.parwork/kernel2.so")
for name in ("k_serial", "k_parallel", "dbg"):
    if hasattr(k2, name):
        getattr(k2, name).argtypes = [ctypes.c_int64] * 10
        getattr(k2, name).restype = ctypes.c_int64

rng = np.random.default_rng(42)
n = 5
R = [rng.integers(0, 200, size=10).astype(np.int64) for _ in range(n)]
H = [r.copy() for r in R]
ro = np.concatenate([[0], np.cumsum([len(r) for r in R])]).astype(np.int64)
ho = np.concatenate([[0], np.cumsum([len(h) for h in H])]).astype(np.int64)
ref = np.concatenate(R)
hyp = np.concatenate(H)
mws = 1
mso = 196
stride = 198
masks = np.zeros(stride * 4, dtype=np.uint64)
got = np.zeros(n, dtype=np.int64)
for count in (0, 1, 2, 5):
    print("count", count, flush=True)
    r = k2.k_serial(ref.ctypes.data, hyp.ctypes.data, ro.ctypes.data, ho.ctypes.data,
                    got.ctypes.data, masks.ctypes.data, stride, mso, count, 1)
    print("returned", r, got[:count].tolist(), flush=True)

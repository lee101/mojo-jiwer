import ctypes
import os
import sys

import numpy as np

base = ctypes.CDLL("/nvme0n1-disk/code/mojo-jiwer/dist/libmojo-jiwer.so")
base.mji_distances.argtypes = [ctypes.c_int64] * 12


def stage(msg):
    print(msg, flush=True)


rng = np.random.default_rng(42)
n = 5
R = [rng.integers(0, 200, size=10).astype(np.int64) for _ in range(n)]
H = [r.copy() for r in R]
ro = np.concatenate([[0], np.cumsum([len(r) for r in R])]).astype(np.int64)
ho = np.concatenate([[0], np.cumsum([len(h) for h in H])]).astype(np.int64)
ref = np.concatenate(R)
hyp = np.concatenate(H)
mws = max(1, (int(np.diff(ho).max()) + 63) // 64)
mso = (int(max(ref.max(), hyp.max())) + 1) * mws
stride = mso + 2 * mws
stage(f"mws={mws} mso={mso} stride={stride} lens={[len(r) for r in R]}")
masks = np.zeros(stride, dtype=np.uint64)
expected = np.zeros(n, dtype=np.int64)
stage("calling baseline mji_distances")
base.mji_distances(ref.ctypes.data, hyp.ctypes.data, ro.ctypes.data, ho.ctypes.data,
                   n, 1, 0, masks.ctypes.data, 0, mws, mso, expected.ctypes.data)
stage(f"baseline {expected.tolist()}")

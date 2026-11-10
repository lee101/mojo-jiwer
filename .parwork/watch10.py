import ctypes

import numpy as np

k2 = ctypes.CDLL("/nvme0n1-disk/code/mojo-jiwer/.parwork/kernel2.so")
base = ctypes.CDLL("/nvme0n1-disk/code/mojo-jiwer/.parwork/baseline.so")
k2.k_serial.argtypes = [ctypes.c_int64] * 10
k2.k_serial.restype = ctypes.c_int64
base.mji_distance.argtypes = [ctypes.c_int64] * 7
base.mji_distance.restype = ctypes.c_int64

rng = np.random.default_rng(0)
for trial in range(5):
    n = 3
    R = [rng.integers(0, 30, size=20).astype(np.int64) for _ in range(n)]
    H = [r + 1 for r in R]
    ro = np.concatenate([[0], np.cumsum([len(r) for r in R])]).astype(np.int64)
    ho = np.concatenate([[0], np.cumsum([len(h) for h in H])]).astype(np.int64)
    ref = np.concatenate(R)
    hyp = np.concatenate(H)
    mws = max(1, (int(np.diff(ho).max()) + 63) // 64)
    mso = (int(max(ref.max(), hyp.max())) + 1) * mws
    stride = mso + 2 * mws
    masks = np.zeros(stride * 2, dtype=np.uint64)
    res = np.zeros(n, dtype=np.int64)
    k2.k_serial(ref.ctypes.data, hyp.ctypes.data, ro.ctypes.data, ho.ctypes.data,
                res.ctypes.data, masks.ctypes.data, stride, mso, n, 1)
    outs = [base.mji_distance(ref.ctypes.data + 8 * int(ro[i]), hyp.ctypes.data + 8 * int(ho[i]),
                              int(ro[i + 1] - ro[i]), int(ho[i + 1] - ho[i]),
                              masks.ctypes.data, mws, mso) for i in range(n)]
    print(trial, "k_serial", res.tolist(), "mji_distance", outs, flush=True)

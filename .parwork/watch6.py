import ctypes
import random

import numpy as np

k2 = ctypes.CDLL("/nvme0n1-disk/code/mojo-jiwer/.parwork/kernel2.so")
k2.k_serial.argtypes = [ctypes.c_int64] * 10
k2.k_serial.restype = ctypes.c_int64


def case(R, H, tag):
    n = len(R)
    ro = np.zeros(n + 1, dtype=np.int64)
    ho = np.zeros(n + 1, dtype=np.int64)
    ro[1:] = np.cumsum([len(r) for r in R])
    ho[1:] = np.cumsum([len(h) for h in H])
    ref = np.array([t for r in R for t in r], dtype=np.int64)
    hyp = np.array([t for h in H for t in h], dtype=np.int64)
    mws = max(1, (int(np.diff(ho).max()) + 63) // 64)
    mso = (max(int(ref.max()), int(hyp.max())) + 1) * mws
    stride = mso + 2 * mws
    masks = np.zeros(stride * 2, dtype=np.uint64)
    res = np.zeros(n, dtype=np.int64)
    r0 = ref.copy()
    h0 = hyp.copy()
    k2.k_serial(ref.ctypes.data, hyp.ctypes.data, ro.ctypes.data, ho.ctypes.data,
                res.ctypes.data, masks.ctypes.data, stride, mso, n, 1)
    print(tag, "lens", [len(r) for r in R], "stride", stride, "mso", mso,
          "refdiff", int((ref != r0).sum()), "hypdiff", int((hyp != h0).sum()),
          "refidx", np.nonzero(ref != r0)[0][:6], "hypidx", np.nonzero(hyp != h0)[0][:6],
          "res", res, flush=True)


rng = random.Random(3)
for trial in range(10):
    n = rng.randint(1, 4)
    R, H = [], []
    for _ in range(n):
        L = rng.randint(1, 30)
        r = [rng.randrange(30) for _ in range(L)]
        h = r.copy()
        for i in range(0, L, 9):
            h[i] = rng.randrange(30)
        R.append(r)
        H.append(h)
    if trial == 9:
        case(R, H, "trial9")
        print("R", R, "H", H, flush=True)

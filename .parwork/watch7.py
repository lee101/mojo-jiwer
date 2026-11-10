import ctypes

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
    mso = 252 * mws
    stride = mso + 2 * mws
    masks = np.zeros(stride * 2 + 4, dtype=np.uint64)
    res = np.zeros(n, dtype=np.int64)
    r0 = ref.copy()
    k2.k_serial(ref.ctypes.data, hyp.ctypes.data, ro.ctypes.data, ho.ctypes.data,
                res.ctypes.data, masks.ctypes.data, stride, mso, n, 1)
    bad = int((ref != r0).sum())
    if bad:
        print(tag, "lens", [len(r) for r in R], "stride", stride, "mso", mso,
              "refidx", np.nonzero(ref != r0)[0][:6],
              "newvals", ref[np.nonzero(ref != r0)[0][:6]], "res", res, flush=True)
    return bad


R = [[28, 19, 16, 13, 17, 29, 22, 7, 1, 23, 14, 26, 24, 21, 23, 16, 9, 17, 10, 28, 7, 27, 2, 27, 18, 9, 3],
     [28, 25]]
H = [[25, 19, 16, 13, 17, 29, 22, 7, 1, 7, 14, 26, 24, 21, 23, 16, 9, 17, 1, 28, 7, 27, 2, 27, 18, 9, 3],
     [22, 25]]
case(R, H, "pair0")
case([R[1]], [H[1]], "pair1")
case([R[0]], [H[0]], "only0")
print("--- single short pair, sweep")
for a in range(5):
    for b in range(5):
        case([[a, b]], [[(a + 1) % 5, b]], f"{a}{b}")

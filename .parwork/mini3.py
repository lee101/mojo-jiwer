import ctypes

import numpy as np

base = ctypes.CDLL("/nvme0n1-disk/code/mojo-jiwer/dist/libmojo-jiwer.so")
k2 = ctypes.CDLL("/nvme0n1-disk/code/mojo-jiwer/.parwork/kernel2.so")
base.mji_distances.argtypes = [ctypes.c_int64] * 12
k2.k_serial.argtypes = [ctypes.c_int64] * 10
k2.k_serial.restype = ctypes.c_int64

rng = np.random.default_rng(42)
trial = 0
n = int(rng.integers(1, 60))
R = [rng.integers(0, 200, size=int(rng.integers(1, 90))).astype(np.int64) for _ in range(n)]
H = []
for r in R:
    h = r.copy()
    k = max(1, len(h) // 7)
    for i in range(0, len(h), k):
        h[i] = int(rng.integers(0, 200))
    H.append(h)
print("n", n, "lens", [len(r) for r in R], flush=True)
for start in range(0, n, 1):
    sub_R = R[start:start + 1]
    sub_H = H[start:start + 1]
    ro = np.concatenate([[0], np.cumsum([len(r) for r in sub_R])]).astype(np.int64)
    ho = np.concatenate([[0], np.cumsum([len(h) for h in sub_H])]).astype(np.int64)
    ref = np.concatenate(sub_R)
    hyp = np.concatenate(sub_H)
    mws = max(1, (int(np.diff(ho).max()) + 63) // 64)
    mso = (int(max(ref.max(), hyp.max())) + 1) * mws
    stride = mso + 2 * mws
    masks = np.zeros(stride, dtype=np.uint64)
    got = np.zeros(1, dtype=np.int64)
    print(f"pair {start}: ref_len={len(ref)} hyp_len={len(hyp)} mws={mws} mso={mso} stride={stride} "
          f"maxref={int(ref.max())} maxhyp={int(hyp.max())}", flush=True)
    k2.k_serial(ref.ctypes.data, hyp.ctypes.data, ro.ctypes.data, ho.ctypes.data,
                got.ctypes.data, masks.ctypes.data, stride, mso, 1, 1)
    print("  ok", got.tolist(), flush=True)

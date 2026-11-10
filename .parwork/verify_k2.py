import ctypes

import numpy as np

base = ctypes.CDLL("/nvme0n1-disk/code/mojo-jiwer/dist/libmojo-jiwer.so")
k2 = ctypes.CDLL("/nvme0n1-disk/code/mojo-jiwer/.parwork/kernel2.so")
base.mji_distances.argtypes = [ctypes.c_int64] * 12
k2.k_serial.argtypes = [ctypes.c_int64] * 10
k2.k_serial.restype = ctypes.c_int64
k2.k_parallel.argtypes = [ctypes.c_int64] * 10
k2.k_parallel.restype = ctypes.c_int64

rng = np.random.default_rng(42)
ok = 0
for trial in range(8):
    n = int(rng.integers(1, 60))
    R = [rng.integers(0, 200, size=int(rng.integers(1, 90))).astype(np.int64) for _ in range(n)]
    H = []
    for r in R:
        h = r.copy()
        k = max(1, len(h) // 7)
        for i in range(0, len(h), k):
            h[i] = int(rng.integers(0, 200))
        H.append(h)
    ro = np.concatenate([[0], np.cumsum([len(r) for r in R])]).astype(np.int64)
    ho = np.concatenate([[0], np.cumsum([len(h) for h in H])]).astype(np.int64)
    ref = np.concatenate(R)
    hyp = np.concatenate(H)
    mws = max(1, (int(np.diff(ho).max()) + 63) // 64)
    mso = (int(max(ref.max(), hyp.max())) + 1) * mws
    stride = mso + 2 * mws
    ref0 = ref.copy()

    expected = np.zeros(n, dtype=np.int64)
    masks = np.zeros(max(stride * mws * 4, 4096), dtype=np.uint64)
    base.mji_distances(ref.ctypes.data, hyp.ctypes.data, ro.ctypes.data, ho.ctypes.data,
                       n, 1, 0, masks.ctypes.data, 0, mws, mso, expected.ctypes.data)

    got = np.zeros(n, dtype=np.int64)
    masks = np.zeros(max(stride * mws * 4, 4096), dtype=np.uint64)
    k2.k_serial(ref.ctypes.data, hyp.ctypes.data, ro.ctypes.data, ho.ctypes.data,
                got.ctypes.data, masks.ctypes.data, stride, mso, n, 1)

    par = np.zeros(n, dtype=np.int64)
    masks = np.zeros(max(stride * mws * 4, 4096), dtype=np.uint64)
    k2.k_parallel(ref.ctypes.data, hyp.ctypes.data, ro.ctypes.data, ho.ctypes.data,
                  par.ctypes.data, masks.ctypes.data, stride, mso, n, 4)
    good = (got == expected).all() and (par == expected).all() and (ref == ref0).all()
    ok += bool(good)
    print(trial, "n", n, "maxlen", int(np.diff(ro).max()), "mws", mws,
          "serial", bool((got == expected).all()), "parallel", bool((par == expected).all()),
          "ref untouched", bool((ref == ref0).all()), flush=True)
print("trials ok", ok, "of 8")

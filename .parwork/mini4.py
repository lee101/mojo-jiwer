import ctypes

import numpy as np

k2 = ctypes.CDLL("/nvme0n1-disk/code/mojo-jiwer/.parwork/kernel2.so")
k2.k_serial.argtypes = [ctypes.c_int64] * 10
k2.k_serial.restype = ctypes.c_int64

rng = np.random.default_rng(42)
R = [rng.integers(0, 200, size=69).astype(np.int64)]
H = [R[0].copy()]
ro = np.array([0, 69], dtype=np.int64)
ho = np.array([0, 69], dtype=np.int64)
ref = R[0].copy()
hyp = H[0].copy()
print("R", ref.tolist())
print("H", hyp.tolist())
n = 1
L = 69
mws = 2
mso_global = (int(max(ref.max(), hyp.max())) + 1) * mws
smallest_mso = (int(max(R[0].max(), H[0].max())) + 1) * mws
print("mws", mws, "mso_global", mso_global, "smallest_mso", smallest_mso)
for name, mso in (("global", mso_global), ("smallest", smallest_mso)):
    for guard in (0, 1):
        stride = mso + 2 * mws + guard * 300
        masks = np.zeros(stride, dtype=np.uint64)
        got = np.zeros(1, dtype=np.int64)
        k2.k_serial(ref.ctypes.data, hyp.ctypes.data, ro.ctypes.data, ho.ctypes.data,
                    got.ctypes.data, masks.ctypes.data, stride, mso, 1, 1)
        print(name, "guard", guard, "stride", stride, "result", got.tolist(),
              "dirty", int((masks != 0).sum()), flush=True)

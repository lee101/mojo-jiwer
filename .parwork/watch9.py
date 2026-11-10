import ctypes

import numpy as np

k2 = ctypes.CDLL("/nvme0n1-disk/code/mojo-jiwer/.parwork/kernel2.so")
k2.k_serial.argtypes = [ctypes.c_int64] * 10
k2.k_serial.restype = ctypes.c_int64

R = [[28, 19, 16, 13, 17, 29, 22, 7, 1, 23, 14, 26, 24, 21, 23, 16, 9, 17, 10, 28, 7, 27, 2, 27, 18, 9, 3],
     [28, 25]]
H = [[25, 19, 16, 13, 17, 29, 22, 7, 1, 7, 14, 26, 24, 21, 23, 16, 9, 17, 1, 28, 7, 27, 2, 27, 18, 9, 3],
     [22, 25]]

n = 2
ro = np.zeros(n + 1, dtype=np.int64)
ho = np.zeros(n + 1, dtype=np.int64)
ro[1:] = [len(r) for r in R]
ho[1:] = [len(h) for h in H]
ref = np.array([t for r in R for t in r], dtype=np.int64)
hyp = np.array([t for h in H for t in h], dtype=np.int64)
stride = 256
mso = 256
print("ref", ref.tolist())
print("hyp", hyp.tolist())
for rep in range(6):
    masks = np.zeros(stride * 2 + 8, dtype=np.uint64)
    res = np.zeros(n, dtype=np.int64)
    k2.k_serial(ref.ctypes.data, hyp.ctypes.data, ro.ctypes.data, ho.ctypes.data,
                res.ctypes.data, masks.ctypes.data, stride, mso, n, 1)
    print(rep, "res", res.tolist(), "dirty mask words", int((masks != 0).sum()),
          "max token", int(max(ref.max(), hyp.max())))

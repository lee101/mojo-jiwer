import ctypes, numpy as np, sys
lib = ctypes.CDLL(sys.argv[1])
lib.probe_items.argtypes=[ctypes.c_int64]*3; lib.probe_items.restype=ctypes.c_int64
for n in (1, 7, 100, 1000, 100000):
    buf=np.full(n,-1,dtype=np.int64)
    r=lib.probe_items(buf.ctypes.data,n,4)
    print(n, r, bool((buf==np.arange(n,dtype=np.int64)*2).all()), flush=True)

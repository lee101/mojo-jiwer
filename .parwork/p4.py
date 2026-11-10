import ctypes, sys, numpy as np
lib = ctypes.CDLL(sys.argv[1])
lib.store_items.argtypes=[ctypes.c_int64]*3; lib.store_items.restype=ctypes.c_int64
mode, n, w = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
buf=np.full(n,-1,dtype=np.int64)
print(lib.store_items(buf.ctypes.data,n,w), flush=True)
print(bool((buf==np.arange(n)).all()), flush=True)

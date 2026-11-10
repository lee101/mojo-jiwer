import ctypes, sys, numpy as np
from concurrent.futures import ThreadPoolExecutor
lib = ctypes.CDLL(sys.argv[1])
lib.store_span.argtypes=[ctypes.c_int64]*4; lib.store_span.restype=ctypes.c_int64
n, w = int(sys.argv[2]), int(sys.argv[3])
buf=np.full(n,-1,dtype=np.int64)
with ThreadPoolExecutor(max_workers=w) as ex:
    list(ex.map(lambda c: lib.store_span(buf.ctypes.data,n,w,c), range(w)))
print(bool((buf==np.arange(n)).all()), flush=True)

import ctypes, sys
lib = ctypes.CDLL(sys.argv[1])
lib.call_workers.argtypes=[ctypes.c_int64]*2; lib.call_workers.restype=ctypes.c_int64
n,w = int(sys.argv[2]), int(sys.argv[3])
print("calling", n, w, flush=True)
print(lib.call_workers(n,w), flush=True)

import ctypes, numpy as np, os, sys
lib=ctypes.CDLL('/nvme0n1-disk/code/mojo-jiwer/.parwork/ksplit.so')
lib.spin_work.argtypes=[ctypes.c_int64]*4; lib.spin_work.restype=ctypes.c_int64
lib.heavy_call.argtypes=[ctypes.c_int64]*5; lib.heavy_call.restype=ctypes.c_int64
n,w,heavy=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])
buf=np.zeros(n,dtype=np.int64)
if os.environ.get('VIA'):
    lib.heavy_call(buf.ctypes.data,n,heavy,w,int(os.environ['VIA']))
else:
    lib.spin_work(buf.ctypes.data,n,heavy,w)
print("ok",flush=True)
os._exit(0)

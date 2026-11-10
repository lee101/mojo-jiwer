import ctypes
import numpy as np
k2 = ctypes.CDLL("/nvme0n1-disk/code/mojo-jiwer/.parwork/kernel2.so")
k2.k_serial.argtypes = [ctypes.c_int64] * 10
k2.k_serial.restype = ctypes.c_int64
ref=np.array([1,2,3],dtype=np.int64); hyp=np.array([1,2,3],dtype=np.int64)
ro=np.array([0,3],dtype=np.int64); ho=np.array([0,3],dtype=np.int64)
masks=np.zeros(64,dtype=np.uint64); got=np.zeros(1,dtype=np.int64)
print("call0", k2.k_serial(ref.ctypes.data,hyp.ctypes.data,ro.ctypes.data,ho.ctypes.data,got.ctypes.data,masks.ctypes.data,16,16,0,1), flush=True)
print("call1", k2.k_serial(ref.ctypes.data,hyp.ctypes.data,ro.ctypes.data,ho.ctypes.data,got.ctypes.data,masks.ctypes.data,16,16,1,1), got.tolist(), flush=True)

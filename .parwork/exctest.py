import ctypes, numpy as np
lib=ctypes.CDLL('/nvme0n1-disk/code/mojo-jiwer/.parwork/exact.so')
lib.mji_distance.argtypes=[ctypes.c_int64]*7; lib.mji_distance.restype=ctypes.c_int64
a=np.array([1,2,3],dtype=np.int64); b=np.array([1,9,3],dtype=np.int64)
masks=np.zeros(64,dtype=np.uint64)
print(lib.mji_distance(a.ctypes.data,b.ctypes.data,3,3,masks.ctypes.data,1,8))

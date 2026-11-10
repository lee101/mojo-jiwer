import ctypes, numpy as np, sys
k2=ctypes.CDLL('/nvme0n1-disk/code/mojo-jiwer/.parwork/kernel2.so')
k2.k_serial.argtypes=[ctypes.c_int64]*10; k2.k_serial.restype=ctypes.c_int64
ref=np.array([10,20,30],dtype=np.int64)
ro=np.array([0,3],dtype=np.int64)
hyp=np.array([10,20,31],dtype=np.int64)
ho=np.array([0,3],dtype=np.int64)
res=np.zeros(5,dtype=np.int64)
stride=64; mso=64
masks=np.zeros(stride*2,dtype=np.uint64)
print("ref addr", hex(ref.ctypes.data), flush=True)
r=k2.k_serial(ref.ctypes.data,hyp.ctypes.data,ro.ctypes.data,ho.ctypes.data,res.ctypes.data,masks.ctypes.data,stride,mso,1,1)
print(r,res[:2],ref,hyp, flush=True)

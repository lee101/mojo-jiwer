import ctypes, numpy as np
lib=ctypes.CDLL('/nvme0n1-disk/code/mojo-jiwer/.parwork/echo.so')
lib.echo_args.argtypes=[ctypes.c_int64]*10; lib.echo_args.restype=ctypes.c_int64
res=np.zeros(9,dtype=np.int64)
args=list(range(100,110))
lib.echo_args(res.ctypes.data,*args)
print("sent",args)
print("got ",res.tolist())

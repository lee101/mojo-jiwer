import ctypes, sys, numpy as np
ROOT="/nvme0n1-disk/code/mojo-jiwer"; sys.path.insert(0, ROOT+"/python")
from mojo_jiwer import _native
I64=ctypes.c_int64; A=lambda x:int(x.ctypes.data)
def prep(path):
    lib=ctypes.CDLL(path); lib.mji_distances.argtypes=[I64]*12
    pattern=list(range(1,66)); text=pattern+[1]
    ref=np.array(pattern,dtype=np.int64); hyp=np.array(text,dtype=np.int64)
    ro=np.array([0,65],dtype=np.int64); ho=np.array([0,66],dtype=np.int64)
    masks=np.zeros(68,dtype=np.uint64); out=np.empty(1,dtype=np.int64)
    return lambda: lib.mji_distances(A(ref),A(hyp),A(ro),A(ho),1,1,0,A(masks),0,1,66,A(out))
f=prep(sys.argv[1]); f()
print("ok")

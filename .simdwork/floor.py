import ctypes, sys, resource, time
ROOT="/nvme0n1-disk/code/mojo-jiwer"; sys.path.insert(0, ROOT+"/python")
import numpy as np
I64=ctypes.c_int64; A=lambda x:int(x.ctypes.data)
def cpu():
    r=resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime+r.ru_stime
l=ctypes.CDLL(ROOT+"/dist/libmojo-jiwer.so"); l.mji_distances.argtypes=[I64]*12
for numel in (1,2):
    pat=np.array([1]*numel,dtype=np.int64); txt=np.array([1],dtype=np.int64)
    ro=np.array([0,0],dtype=np.int64); ho=np.array([0,1],dtype=np.int64)
    masks=np.zeros(254,dtype=np.uint64); out=np.empty(1,dtype=np.int64)
    f=lambda: l.mji_distances(A(pat),A(txt),A(ro),A(ho),1,1,0,A(masks),0,1,252,A(out))
    f(); best=float("inf")
    for _ in range(15):
        c=cpu()
        for _ in range(500): f()
        best=min(best,(cpu()-c)/500)
    print(f"mji_distances with count=1 pattern_len=0 (zero utterances of work): {best*1e6:.2f} us CPU")

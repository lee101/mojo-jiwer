import ctypes, sys, resource
ROOT="/nvme0n1-disk/code/mojo-jiwer"; sys.path.insert(0, ROOT+"/python")
import numpy as np
I64=ctypes.c_int64; A=lambda x:int(x.ctypes.data)
def cpu():
    r=resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime+r.ru_stime
lib=ctypes.CDLL(ROOT+"/dist/libmojo-jiwer.so")
lib.mji_distances.argtypes=[I64]*12
pat=np.array([1,2],dtype=np.int64); txt=np.array([1],dtype=np.int64)
ro=np.array([0,0],dtype=np.int64); ho=np.array([0,1],dtype=np.int64)
out=np.empty(1,dtype=np.int64)
def bench(fill):
    masks=np.zeros(1<<20, dtype=np.uint64)
    def f():
        lib.mji_distances(A(pat),A(txt),A(ro),A(ho),1,1,0,A(masks),0,1,10,A(out))
    f(); best=float("inf")
    for _ in range(10):
        c=cpu()
        for _ in range(50): f()
        best=min(best,(cpu()-c)/50)
    return best
for name in ("cold 8 MiB np.zeros (first touch of every page)",):
    t=bench(None)
    print(f"{name}: {t*1e3:.2f} ms per 'no work' call")
# same buffer, second call onward (pages already resident)
masks=np.zeros(1<<20, dtype=np.uint64)
f=lambda: lib.mji_distances(A(pat),A(txt),A(ro),A(ho),1,1,0,A(masks),0,1,10,A(out))
f(); best=float("inf")
for _ in range(10):
    c=cpu()
    for _ in range(500): f()
    best=min(best,(cpu()-c)/500)
print(f"resident 8 MiB buffer, no kernel work: {best*1e6:.2f} us per call")

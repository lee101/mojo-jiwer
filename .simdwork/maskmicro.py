"""CPU-time A/B of the mask build. Measures process CPU time via getrusage and
takes the minimum over many interleaved batches; the kernel is short and the
work is single-threaded, so CPU time is far less sensitive to co-tenant load."""
import ctypes, sys, time, resource
ROOT="/nvme0n1-disk/code/mojo-jiwer"
sys.path.insert(0, ROOT+"/python")
import numpy as np
from mojo_jiwer import _native
I64=ctypes.c_int64; A=lambda x:int(x.ctypes.data)
def load(p):
    l=ctypes.CDLL(p); l.mji_distances.argtypes=[I64]*12; l.mji_distances.restype=None; return l
def cpu():
    r=resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime+r.ru_stime
def make(lib, pat_len, vocab):
    pat=(np.arange(pat_len,dtype=np.int64)%vocab)+1
    txt=pat[:1]
    ro=np.array([0,pat_len],dtype=np.int64); ho=np.array([0,1],dtype=np.int64)
    mws=1; mso=int(pat.max())+1
    masks=np.zeros(mso+2*mws,dtype=np.uint64); out=np.empty(1,dtype=np.int64)
    return lambda reps: [lib.mji_distances(A(pat),A(txt),A(ro),A(ho),1,1,0,A(masks),0,mws,mso,A(out)) for _ in range(reps)]
if __name__=="__main__":
    la,lb=load(sys.argv[1]),load(sys.argv[2])
    print("| pattern | vocab | A us/call | B us/call | B/A |")
    print("|---|---:|---:|---:|---:|")
    for pat_len,vocab in ((200,200),(600,600),(1000,1000),(1000,251),(2000,251)):
        fa,fb=make(la,pat_len,vocab),make(lb,pat_len,vocab)
        reps=2000
        fa(50); fb(50)              # warm
        ta=tb=float("inf")
        for _ in range(40):
            c0=cpu(); fa(reps); ta=min(ta,cpu()-c0)
            c0=cpu(); fb(reps); tb=min(tb,cpu()-c0)
        print(f"| {pat_len} | {vocab} | {ta/reps*1e6:.3f} | {tb/reps*1e6:.3f} | {tb/ta:.3f} |")

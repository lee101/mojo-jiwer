import ctypes, sys, resource
ROOT="/nvme0n1-disk/code/mojo-jiwer"; sys.path.insert(0, ROOT+"/python")
import numpy as np
I64=ctypes.c_int64; A=lambda x:int(x.ctypes.data)
def cpu():
    r=resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime+r.ru_stime
def lib():
    l=ctypes.CDLL(ROOT+"/dist/libmojo-jiwer.so"); l.mji_distances.argtypes=[I64]*12; return l
l=lib()
def bench(f,reps=300,rounds=15):
    f(); best=float("inf")
    for _ in range(rounds):
        c=cpu()
        for _ in range(reps): f()
        best=min(best,(cpu()-c)/reps)
    return best
def full(pat_len,text_len=1):
    pat=(np.arange(pat_len,dtype=np.int64)%251)+1; txt=pat[:text_len]
    ro=np.array([0,pat_len],dtype=np.int64); ho=np.array([0,text_len],dtype=np.int64)
    masks=np.zeros(254,dtype=np.uint64); out=np.empty(1,dtype=np.int64)
    return lambda: l.mji_distances(A(pat),A(txt),A(ro),A(ho),1,1,0,A(masks),0,1,252,A(out))
def draft_clear(pat_len,stride=252):
    pat=(np.arange(pat_len,dtype=np.int64)%251)+1
    ro=np.array([0,pat_len],dtype=np.int64); ho=np.array([0,1],dtype=np.int64)
    masks=np.zeros(254,dtype=np.uint64); out=np.empty(1,dtype=np.int64); txt=pat[:1]
    F=l.mji_distances
    def run():
        # current clear + build, no kernel
        for k in range(0,pat_len,16):
            blk=pat_len-k
            for j in range(blk):
                F  # noop ref
        return None
    return run
# isolate: clear+build cost = full(pat,1) - a call with pattern_len=1 at same stride
base=bench(full(1,1))
for plen in (500,1000,2000):
    t=bench(full(plen,1))
    print(f"pattern {plen:5d}: full {t*1e6:6.2f} us, empty-pattern call {base*1e6:6.2f} us -> mask work {(t-base)*1e6:6.2f} us "
          f"({(t-base)/plen*1e3:.1f} ns/word)")

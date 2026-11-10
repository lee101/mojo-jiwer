import ctypes, sys, resource
ROOT="/nvme0n1-disk/code/mojo-jiwer"; sys.path.insert(0, ROOT+"/python")
import numpy as np
I64=ctypes.c_int64; A=lambda x:int(x.ctypes.data)
def cpu():
    r=resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime+r.ru_stime
def load(p):
    l=ctypes.CDLL(p); 
    for n in ("mji_distances","mji_traces"):
        f=getattr(l,n); f.argtypes=[I64]*12; f.restype=None
    return l
def batch(lib, kind, count, length, vocab, mm, shift):
    rng=np.random.default_rng(0)
    refs=(rng.integers(0,vocab,(count,length))%max(1,mm)).astype(np.int64)
    hyps=refs.copy()
    # clean adjacent pair mismatch + a far mismatch: prefix ~1 word, suffix ~0
    hyps[:,1]=(hyps[:,1]+1)%max(1,mm)
    hyps[:,-1]=(hyps[:,-1]+2)%max(1,mm)
    ro=np.arange(0,(count+1)*length,length,dtype=np.int64)
    ho=ro.copy()
    ref=refs.reshape(-1); hyp=hyps.reshape(-1)
    if kind=="dist":
        masks=np.zeros((max(mm,1)+1)*(1)+4,dtype=np.uint64); out=np.empty(count,dtype=np.int64)
        return lambda: lib.mji_distances(A(ref),A(hyp),A(ro),A(ho),count,1,0,A(masks),0,1,max(mm,1)+1,A(out))
    mats=np.empty((length+1)**2,dtype=np.int32)
    oo=np.arange(0,(count+1)*2*length,2*length,dtype=np.int64)
    ops=np.empty(2*count*length,dtype=np.uint8); cnt=np.empty(count,dtype=np.int64)
    return lambda: lib.mji_traces(A(ref),A(hyp),A(ro),A(ho),count,1,0,A(mats),0,A(oo),A(ops),A(cnt))
def bench(f,reps=6,rounds=10):
    f(); best=float("inf")
    for _ in range(rounds):
        c=cpu()
        for _ in range(reps): f()
        best=min(best,(cpu()-c)/reps)
    return best
a=load(ROOT+"/.simdwork/baseline.so"); bb=load(ROOT+"/.simdwork/G.so")
print("| loop | case | A us | B us | B/A |"); print("|---|---|---:|---:|---:|")
for label,kind,count,length,vocab,mm in [
    ("dist 10k x 12w, 200 token ids","dist",10_000,12,200,45),
    ("dist 10k x 12w, 256 token ids","dist",10_000,12,256,64),
    ("dist 2k x 60w, 200 token ids","dist",2_000,60,200,45),
    ("dist 2k x 60w, 256 token ids","dist",2_000,60,256,64),
    ("traces 2k x 20w, 200 token ids","trace",2_000,20,200,45),
    ("traces 2k x 20w, 256 token ids","trace",2_000,20,256,64)]:
    fa=batch(a,kind,count,length,vocab,mm,0); fb=batch(bb,kind,count,length,vocab,mm,0)
    ta,tb=bench(fa),bench(fb)
    print(f"| {label} | {kind} | {ta*1e6:.2f} | {tb*1e6:.2f} | {ta/tb:.3f} |")

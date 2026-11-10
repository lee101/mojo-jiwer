import ctypes, sys, resource
ROOT="/nvme0n1-disk/code/mojo-jiwer"; sys.path.insert(0, ROOT+"/python"); sys.path.insert(0, ROOT+"/bench")
import numpy as np, bench as b
from mojo_jiwer import _native
from mojo_jiwer.process import _default_transform, _encode
from mojo_jiwer.transformations import wer_default
I64=ctypes.c_int64; A=lambda x:int(x.ctypes.data)
def cpu():
    r=resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime+r.ru_stime
def load(p):
    l=ctypes.CDLL(p); l.mji_traces.argtypes=[I64]*12; l.mji_traces.restype=None; return l
def enc(seed,count,length):
    refs,hyps=b.word_corpus(count,length,seed)
    return _encode(_default_transform(refs,wer_default,True),_default_transform(hyps,wer_default,False))
def make(lib,a):
    ref,hyp,ro,ho=_native._flatten_sequences(a[0],a[1]); rl,hl=np.diff(ro),np.diff(ho); count=len(ro)-1
    stride=(int(rl.max())+1)*(int(hl.max())+1); mats=np.empty(stride,dtype=np.int32)
    oo=np.empty(count+1,dtype=np.int64); oo[0]=0; np.cumsum(rl+hl,out=oo[1:])
    ops=np.empty(max(1,int(oo[-1])),dtype=np.uint8); cnt=np.empty(count,dtype=np.int64)
    return lambda reps: [lib.mji_traces(A(ref),A(hyp),A(ro),A(ho),count,1,0,A(mats),0,A(oo),A(ops),A(cnt)) for _ in range(reps)]
cases=[("traces 2k x 20w",enc(4,2_000,20)),("traces 2k x 60w",enc(2,2_000,60)),("traces 10k x 12w",enc(1,10_000,12))]
la,lb=load(sys.argv[1]),load(sys.argv[2])
print("| case | A us/call | B us/call | B/A |"); print("|---|---:|---:|---:|")
for label,a in cases:
    fa,fb=make(la,a),make(lb,a); reps=20
    fa(reps);fb(reps); ta=tb=float("inf")
    for _ in range(25):
        c=cpu(); fa(reps); ta=min(ta,cpu()-c)
        c=cpu(); fb(reps); tb=min(tb,cpu()-c)
    print(f"| {label} | {ta/reps*1e6:.1f} | {tb/reps*1e6:.1f} | {tb/ta:.3f} |")

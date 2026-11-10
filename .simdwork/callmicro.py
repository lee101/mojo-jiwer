import ctypes, sys, resource
ROOT="/nvme0n1-disk/code/mojo-jiwer"; sys.path.insert(0, ROOT+"/python"); sys.path.insert(0, ROOT+"/bench")
import numpy as np, bench as b
from mojo_jiwer import _native
from mojo_jiwer.process import _default_transform, _encode
from mojo_jiwer.transformations import wer_default, cer_default
I64=ctypes.c_int64; A=lambda x:int(x.ctypes.data)
def load(p):
    l=ctypes.CDLL(p); l.mji_distances.argtypes=[I64]*12; l.mji_distances.restype=None; return l
def cpu():
    r=resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime+r.ru_stime
def flat(a):
    pairs=[(x,y) if len(y)<=len(x) else (y,x) for x,y in zip(a[0],a[1])]
    return _native._flatten_sequences([p[0] for p in pairs],[p[1] for p in pairs])
def make(lib, a):
    ref,hyp,ro,ho=flat(a); rl,hl=np.diff(ro),np.diff(ho); count=len(ro)-1
    short=int(min(rl.max(),hl.max())); mws=max(1,(short+63)//64)
    swap=int(hl.max())>int(rl.max()); mso=((int(hyp.max()) if swap else int(ref.max()))+1)*mws
    stride=mso+2*mws; masks=np.zeros(stride,dtype=np.uint64); out=np.empty(count,dtype=np.int64)
    return lambda reps: [lib.mji_distances(A(ref),A(hyp),A(ro),A(ho),count,1,0,A(masks),0,mws,mso,A(out)) for _ in range(reps)]
def enc(seed,count,length,kind):
    refs,hyps=(b.word_corpus(count,length,seed) if kind=="wer" else b.character_corpus(count,length,seed))
    t=wer_default if kind=="wer" else cer_default
    return _encode(_default_transform(refs,t,True),_default_transform(hyps,t,False))
cases=[("WER 10k x 12w",enc(1,10_000,12,"wer")),("WER 2k x 60w",enc(2,2_000,60,"wer")),
       ("CER 5k x 48c",enc(3,5_000,48,"cer")),("q 2k x 20w",enc(4,2_000,20,"wer")),
       ("WER 1 x 1000w",enc(5,1,1_000,"wer"))]
la,lb=load(sys.argv[1]),load(sys.argv[2])
print("| case | pairs | A us/call | B us/call | B/A |")
print("|---|---:|---:|---:|---:|")
for label,a in cases:
    fa,fb=make(la,a),make(lb,a); reps=200 if len(a[0])>5000 else 2000
    fa(reps//4); fb(reps//4); ta=tb=float("inf")
    for _ in range(30):
        c=cpu(); fa(reps); ta=min(ta,cpu()-c)
        c=cpu(); fb(reps); tb=min(tb,cpu()-c)
    print(f"| {label} | {len(a[0])} | {ta/reps*1e6:.3f} | {tb/reps*1e6:.3f} | {tb/ta:.3f} |")

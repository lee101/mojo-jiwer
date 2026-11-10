"""Kernel-only interleaved A/B at real caller sizes, min-of-N (min rejects the
co-tenant stalls far better than any average). Prints run arrays + medians."""
import ctypes, sys, statistics
ROOT="/nvme0n1-disk/code/mojo-jiwer"; sys.path.insert(0, ROOT+"/python"); sys.path.insert(0, ROOT+"/bench")
import numpy as np, bench as b
from mojo_jiwer import _native
from mojo_jiwer.process import _default_transform, _encode
from mojo_jiwer.transformations import wer_default, cer_default
I64=ctypes.c_int64; A=lambda x:int(x.ctypes.data)
def load(p):
    l=ctypes.CDLL(p)
    for n in ("mji_distances","mji_traces"):
        f=getattr(l,n); f.argtypes=[I64]*12; f.restype=None
    return l
def layout(rl,hl,ref,hyp):
    short=int(min(rl.max(initial=0),hl.max(initial=0))); mws=max(1,(short+63)//64)
    swap=int(hl.max(initial=0))>int(rl.max(initial=0))
    mso=((int(hyp.max()) if swap else int(ref.max()))+1)*mws
    return mws,mso,mso+2*mws
def dist(lib,a):
    pairs=[(x,y) if len(y)<=len(x) else (y,x) for x,y in zip(a[0],a[1])]
    ref,hyp,ro,ho=_native._flatten_sequences([p[0] for p in pairs],[p[1] for p in pairs])
    rl,hl=np.diff(ro),np.diff(ho); count=len(ro)-1; mws,mso,stride=layout(rl,hl,ref,hyp)
    masks=np.zeros(stride,dtype=np.uint64); out=np.empty(count,dtype=np.int64)
    return (lambda: lib.mji_distances(A(ref),A(hyp),A(ro),A(ho),count,1,0,A(masks),0,mws,mso,A(out))), out
def trc(lib,a):
    ref,hyp,ro,ho=_native._flatten_sequences(a[0],a[1]); rl,hl=np.diff(ro),np.diff(ho); count=len(ro)-1
    mats=np.empty((int(rl.max())+1)*(int(hl.max())+1),dtype=np.int32)
    oo=np.empty(count+1,dtype=np.int64); oo[0]=0; np.cumsum(rl+hl,out=oo[1:])
    ops=np.zeros(max(1,int(oo[-1])),dtype=np.uint8); cnt=np.empty(count,dtype=np.int64)
    return (lambda: lib.mji_traces(A(ref),A(hyp),A(ro),A(ho),count,1,0,A(mats),0,A(oo),A(ops),A(cnt))), (ops,cnt)
def enc(seed,count,length,kind):
    refs,hyps=(b.word_corpus(count,length,seed) if kind=="wer" else b.character_corpus(count,length,seed))
    t=wer_default if kind=="wer" else cer_default
    return _encode(_default_transform(refs,t,True),_default_transform(hyps,t,False))
def mn(f,reps,rounds):
    f(); best=float("inf")
    for _ in range(rounds):
        import time; s=time.perf_counter()
        for _ in range(reps): f()
        best=min(best,(time.perf_counter()-s)/reps)
    return best
old,new=load(sys.argv[1]),load(sys.argv[2]); runs=int(sys.argv[3]) if len(sys.argv)>3 else 12
cases=[("mji_distances, WER 10k x 12w",enc(1,10_000,12,"wer"),dist,4),
       ("mji_distances, WER 2k x 60w",enc(2,2_000,60,"wer"),dist,4),
       ("mji_distances, CER 5k x 48c",enc(3,5_000,48,"cer"),dist,4),
       ("mji_distances, 2k x 20w",enc(4,2_000,20,"wer"),dist,8),
       ("mji_distances, WER 1 x 1000w",enc(5,1,1_000,"wer"),dist,20),
       ("mji_traces, 2k x 20w",enc(4,2_000,20,"wer"),trc,4)]
print("| loop | run | old us | new us | old/new |"); print("|---|---|---:|---:|---:|")
med={}
for label,a,mk,reps in cases:
    fo,co=mk(old,a); fnew,cnew=mk(new,a); fo(); fnew()
    assert ([int(v) for v in (co if not isinstance(co,tuple) else co[0])] ==
            [int(v) for v in (cnew if not isinstance(cnew,tuple) else cnew[0])]), label
    rs=[]
    for r in range(runs):
        to=mn(fo,reps,3); tn=mn(fnew,reps,3)
        rs.append(to/tn)
        print(f"| {label} | {r+1} | {to*1e6:.2f} | {tn*1e6:.2f} | {to/tn:.3f} |")
    med[label]=rs
print(); print("| loop | median old/new | min | max |"); print("|---|---:|---:|---:|")
for k,v in med.items(): print(f"| {k} | {statistics.median(v):.3f} | {min(v):.3f} | {max(v):.3f} |")

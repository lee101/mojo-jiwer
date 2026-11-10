import ctypes, sys, resource, statistics
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
def make(lib, ref, hyp, ro, ho, count):
    rl,hl=np.diff(ro),np.diff(ho)
    mats=np.empty((int(rl.max())+1)*(int(hl.max())+1),dtype=np.int32)
    oo=np.empty(count+1,dtype=np.int64); oo[0]=0; np.cumsum(rl+hl,out=oo[1:])
    ops=np.empty(max(1,int(oo[-1])),dtype=np.uint8); cnt=np.empty(count,dtype=np.int64)
    return lambda reps: [lib.mji_traces(A(ref),A(hyp),A(ro),A(ho),count,1,0,A(mats),0,A(oo),A(ops),A(cnt)) for _ in range(reps)], (ops,cnt)
def interleave(fa,fb,reps,rounds=12):
    fa(reps); fb(reps); rs=[]
    for _ in range(rounds):
        c=cpu(); fa(reps); ta=cpu()-c
        c=cpu(); fb(reps); tb=cpu()-c
        rs.append(ta/tb)
    return statistics.median(rs),min(rs),max(rs)
old,new=load(sys.argv[1]),load(sys.argv[2])
print("| case | median A/B | min | max |"); print("|---|---:|---:|---:|")
for label,(r,h),count in [("real 2k x 20w",b.word_corpus(2_000,20,4),2_000),
                          ("real 10k x 12w",b.word_corpus(10_000,12,1),10_000),
                          ("real 2k x 60w",b.word_corpus(2_000,60,2),2_000)]:
    rt=_default_transform(r,wer_default,True); ht=_default_transform(h,wer_default,False); a=_encode(rt,ht)
    ref,hyp,ro,ho=_native._flatten_sequences(a[0],a[1])
    fa,ca=make(old,ref,hyp,ro,ho,count); fb,cb=make(new,ref,hyp,ro,ho,count)
    # each library writes into its own zeroed buffer once; compare all of it
    ca[0][:]=0; fa(1); snap_old=ca[0].copy(); cnt_old=ca[1].copy()
    cb[0][:]=0; fb(1); snap_new=cb[0].copy(); cnt_new=cb[1].copy()
    assert snap_old.tolist()==snap_new.tolist(), f"{label}: op buffers differ"
    assert cnt_old.tolist()==cnt_new.tolist(), f"{label}: counts differ"
    reps= 6 if count>=10_000 else 12
    m,lo,hi=interleave(fa,fb,reps)
    print(f"| {label} | {m:.3f} | {lo:.3f} | {hi:.3f} |")
# adversarial: clean 12-word pairs (extra probes cut the prefix at word 1)
rng=np.random.default_rng(0); count=10_000; length=12
refs=(rng.integers(0,45,(count,length))).astype(np.int64); hyps=refs.copy()
hyps[:,1]=(hyps[:,1]+1)%45; hyps[:,-1]=(hyps[:,-1]+2)%45
ro=np.arange(0,(count+1)*length,length,dtype=np.int64); ho=ro.copy()
fa,ca=make(old,refs.reshape(-1),hyps.reshape(-1),ro,ho,count); fb,cb=make(new,refs.reshape(-1),hyps.reshape(-1),ro,ho,count)
ca[0][:]=0; fa(1); s1=ca[0].copy()
cb[0][:]=0; fb(1); s2=cb[0].copy()
assert s1.tolist()==s2.tolist(), "adversarial: op buffers differ"
m,lo,hi=interleave(fa,fb,6)
print(f"| clean-prefix 10k x 12w (adversarial) | {m:.3f} | {lo:.3f} | {hi:.3f} |")

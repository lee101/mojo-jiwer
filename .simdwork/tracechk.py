import ctypes, sys
ROOT="/nvme0n1-disk/code/mojo-jiwer"; sys.path.insert(0, ROOT+"/python"); sys.path.insert(0, ROOT+"/bench")
import numpy as np, bench as b
from mojo_jiwer import _native
from mojo_jiwer.process import _default_transform, _encode
from mojo_jiwer.transformations import wer_default
I64=ctypes.c_int64; A=lambda x:int(x.ctypes.data)
def load(p):
    l=ctypes.CDLL(p); l.mji_traces.argtypes=[I64]*12; return l
r,h=b.word_corpus(2_000,20,4)
a=_encode(_default_transform(r,wer_default,True),_default_transform(h,wer_default,False))
ref,hyp,ro,ho=_native._flatten_sequences(a[0],a[1]); count=len(ro)-1
rl,hl=np.diff(ro),np.diff(ho)
oo=np.empty(count+1,dtype=np.int64); oo[0]=0; np.cumsum(rl+hl,out=oo[1:])
outs={}
for tag,path in (("old",ROOT+"/.simdwork/baseline.so"),("new",ROOT+"/.simdwork/H.so")):
    lib=load(path)
    mats=np.empty((int(rl.max())+1)*(int(hl.max())+1),dtype=np.int32)
    ops=np.zeros(max(1,int(oo[-1])),dtype=np.uint8); cnt=np.zeros(count,dtype=np.int64)
    lib.mji_traces(A(ref),A(hyp),A(ro),A(ho),count,1,0,A(mats),0,A(oo),A(ops),A(cnt))
    outs[tag]=(ops.copy(),cnt.copy())
o1,c1=outs["old"]; o2,c2=outs["new"]
d=np.nonzero(o1!=o2)[0]
print("counts equal:", (c1==c2).all(), "| differing op positions:", len(d), d[:10])
if len(d):
    i=int(d[0]); print("first diff at", i, "old", o1[i], "new", o2[i])
    # which sentence?
    idx=int(np.searchsorted(oo, i, side="right")-1); print("sentence", idx, "offset", oo[idx], "len", oo[idx+1]-oo[idx])
    print("old slice", o1[oo[idx]:oo[idx+1]].tolist())
    print("new slice", o2[oo[idx]:oo[idx+1]].tolist())
    print("ref/hyp lens", len(a[0][idx]), len(a[1][idx]))

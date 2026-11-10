import ctypes, sys, resource
ROOT="/nvme0n1-disk/code/mojo-jiwer"; sys.path.insert(0, ROOT+"/python")
import numpy as np
I64=ctypes.c_int64; A=lambda x:int(x.ctypes.data)
def cpu():
    r=resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime+r.ru_stime
lib=ctypes.CDLL(ROOT+"/dist/libmojo-jiwer.so"); lib.mji_distances.argtypes=[I64]*12
def setup(pat_len, text_len):
    pat=(np.arange(pat_len,dtype=np.int64)%251)+1
    txt=pat[:text_len]
    ro=np.array([0,pat_len],dtype=np.int64); ho=np.array([0,text_len],dtype=np.int64)
    mws=1; mso=252
    masks=np.zeros(mso+2*mws,dtype=np.uint64); out=np.empty(1,dtype=np.int64)
    return lambda: lib.mji_distances(A(pat),A(txt),A(ro),A(ho),1,1,0,A(masks),0,mws,mso,A(out))
def bench(f,reps=300,rounds=15):
    f(); best=float("inf")
    for _ in range(rounds):
        c=cpu()
        for _ in range(reps): f()
        best=min(best,(cpu()-c)/reps)
    return best
for plen in (1000,1500,2000,4000,8000):
    t1=bench(setup(plen,1))
    t2=bench(setup(plen,plen))
    print(f"pattern {plen:5d}: mask+1 text word {t1*1e6:7.2f} us | full text {t2*1e6:8.2f} us | "
          f"mask share {t1/t2*100:5.1f}%")
# what the real 1000-word corpus looks like
sys.path.insert(0, ROOT+"/bench")
import bench as b
from mojo_jiwer.process import _default_transform, _encode
from mojo_jiwer.transformations import wer_default
r,h=b.word_corpus(1,1_000,5)
a=_encode(_default_transform(r,wer_default,True),_default_transform(h,wer_default,False))
print("real 1000-word corpus: pattern words", len(a[0][0]), "max token id", max(max(x) for x in a[0]+a[1]))

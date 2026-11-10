import ctypes, sys, resource
ROOT="/nvme0n1-disk/code/mojo-jiwer"; sys.path.insert(0, ROOT+"/python")
import numpy as np
I64=ctypes.c_int64; A=lambda x:int(x.ctypes.data)
def cpu():
    r=resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime+r.ru_stime
lib=ctypes.CDLL(ROOT+"/dist/libmojo-jiwer.so"); lib.mji_distances.argtypes=[I64]*12
def call(pat_len, text_words, reps=400, rounds=20):
    pat=(np.arange(pat_len,dtype=np.int64)%251)+1
    txt=np.resize(pat,text_words) if text_words>1 else pat[:1]
    ro=np.array([0,pat_len],dtype=np.int64); ho=np.array([0,text_words],dtype=np.int64)
    mws=(pat_len+63)//64; mso=252*mws
    masks=np.zeros(mso+2*mws,dtype=np.uint64); out=np.empty(1,dtype=np.int64)
    f=lambda: lib.mji_distances(A(pat),A(txt),A(ro),A(ho),1,1,0,A(masks),0,mws,mso,A(out))
    f(); best=float("inf")
    for _ in range(rounds):
        c=cpu()
        for _ in range(reps): f()
        best=min(best,(cpu()-c)/reps)
    return best
# a 1-word pattern as the fixed-cost reference (same ctypes path, no mask traffic)
for pat_len in (1000, 8000):
    t1=call(pat_len,1); t2=call(pat_len,pat_len)
    print(f"pattern {pat_len}: mask+1 word {t1*1e6:.2f} us | full text {t2*1e6:.2f} us | mask share {(t1/t2)*100:.1f}% | mask alone {(t1-call(1,1))*1e6:.2f} us")

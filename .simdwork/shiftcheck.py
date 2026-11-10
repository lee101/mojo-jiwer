import ctypes, sys
ROOT="/nvme0n1-disk/code/mojo-jiwer"
sys.path.insert(0, ROOT+"/python"); sys.path.insert(0, ROOT+"/bench")
import numpy as np, bench as b
from mojo_jiwer import _native
from mojo_jiwer.process import _default_transform, _encode
from mojo_jiwer.transformations import wer_default
I64=ctypes.c_int64; A=_native._address
lib=ctypes.CDLL(sys.argv[1]); lib.mji_distances.argtypes=[I64]*12
REPS=200
def many_calls(ref, hyp):
    # one utterance whose pattern is repeated within one call (text len 1 work)
    pat=np.array(ref,dtype=np.int64); txt=np.array(hyp,dtype=np.int64)
    ro=np.array([0,len(ref)],dtype=np.int64); ho=np.array([0,len(hyp)],dtype=np.int64)
    mws=1; mso=int(pat.max())+1
    masks=np.zeros(mso+2*mws,dtype=np.uint64); out=np.empty(1,dtype=np.int64)
    def run():
        for _ in range(REPS):
            lib.mji_distances(A(pat),A(txt),A(ro),A(ho),1,1,0,A(masks),0,mws,mso,A(out))
    return run, out
import time
for plen in (200, 600, 1000, 2000):
    pat=list(range(1,plen+1))          # all distinct -> ~all bits set in every slot
    txt=pat[:1]
    run,out=many_calls(pat,txt)
    t=float("inf")
    for _ in range(5):
        s=time.perf_counter(); run(); t=min(t,time.perf_counter()-s)
    print(f"pattern {plen}: {t/REPS*1e6:8.2f} us/call  (dist {int(out[0])})")

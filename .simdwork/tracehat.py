import ctypes, sys
ROOT="/nvme0n1-disk/code/mojo-jiwer"
sys.path.insert(0, ROOT+"/python"); sys.path.insert(0, ROOT+"/bench")
import numpy as np, bench as b
from mojo_jiwer import _native
from mojo_jiwer.process import _default_transform, _encode
from mojo_jiwer.transformations import wer_default
I64=ctypes.c_int64
A=_native._address
def arr(x): return np.array(x,dtype=np.int64)
def trace_hat(lib, ref, hyp, width, flip):
    # emulate trace_impl's row-0 init exactly (old vs new), then scan the
    # operation buffer for multiword-length pairs
    ref=np.array(ref,dtype=np.int64); hyp=np.array(hyp,dtype=np.int64)
    n,m=len(ref),len(hyp)
    count=1
    matrices=np.empty((n+1)*(m+1),dtype=np.int32)
    ops=np.zeros(max(1,n+m),dtype=np.uint8)
    lib.mji_traces(A(ref),A(hyp),A(arr([0,n])),A(arr([0,m])),count,1,0,A(matrices),0,
                   A(arr([0,n+m])),A(ops),A(np.empty(1,dtype=np.int64)))
    # multiword pairs: ops carry a reset artefact for whichever slot a hint
    # token wrote; report the number of non-zero ops and first 8 bytes
    return int((ops!=0).sum()), ops[:8].tolist()
def build(ref, hyp, lib):
    ref=np.array(ref,dtype=np.int64); hyp=np.array(hyp,dtype=np.int64)
    n,m=len(ref),len(hyp)
    mats=np.empty((n+1)*(m+1),dtype=np.int32)
    ops=np.zeros(max(1,n+m),dtype=np.uint8); cnt=np.zeros(1,dtype=np.int64)
    lib.mji_traces(A(ref),A(hyp),A(arr([0,n])),A(arr([0,m])),1,1,0,A(mats),0,
                   A(arr([0,n+m])),A(ops),A(cnt))
    return ops.copy(), cnt.copy()
if __name__=="__main__":
    lib=ctypes.CDLL(sys.argv[1]); lib.mji_traces.argtypes=[I64]*12
    vocab=[f"w{i}" for i in range(250)]
    refs, hyps = b.word_corpus(200, 200, 11)
    diffs=0
    for r,h in zip(refs,hyps):
        rr=_encode(_default_transform([r],wer_default,True),_default_transform([h],wer_default,False))
        a,b_=build(rr[0][0],rr[1][0],lib)
        # an extra field: ops beyond the returned count should be untouched; the
        # artefact is that reset also zeroes slots that only hint tokens use
        diffs+=int(a.any())
    print("pairs whose op buffer saw any write:",diffs,"of",len(refs))

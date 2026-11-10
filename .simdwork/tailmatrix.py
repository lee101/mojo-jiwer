import ctypes, sys, numpy as np
ROOT="/nvme0n1-disk/code/mojo-jiwer"; sys.path.insert(0, ROOT+"/python")
from mojo_jiwer import _native
I64=ctypes.c_int64; A=lambda x:int(x.ctypes.data)
libs={"dist LLVM (baseline)":ROOT+"/.simdwork/baseline.so",
      "dist LLVM (shipped)":ROOT+"/dist/libmojo-jiwer.so"}
BAD=7
def check(path, n, m, start_off):
    lib=ctypes.CDLL(path); lib.mji_trace.argtypes=[I64]*6
    # matrix_trace indexes (n+1) rows x (m+1) columns (n, m include prefix/suffix)
    W=(m+1)*(n+1)
    cells=W+start_off+16
    buf=np.full(cells,BAD,dtype=np.int32)
    view=buf[start_off:start_off+W]
    # tokens all distinct and disjoint so prefix/suffix trim nothing
    ref=(np.arange(n,dtype=np.int64)+1); hyp=(np.arange(m,dtype=np.int64)+1000)
    ops=np.zeros(max(1,n+m),dtype=np.uint8)
    lib.mji_trace(A(ref),A(hyp),n,m,A(view),A(ops))
    # rows 1..n are filled by the DP; row 0 must hold (0..m)
    row0=view[:m+1].copy()
    return row0.tolist()==list(range(m+1)), int((view==BAD).sum())
print("| matrix n x m | start byte offset | row 0 complete | cells left as sentinel |")
print("|---|---|---|---:|")
for pathname,path in libs.items():
    for (n,m) in ((16,17),(20,20),(40,40),(63,64),(64,65),(100,101)):
        for off in (0,4,8,12):
            ok,missing=check(path,n,m,off)
            print(f"| {pathname}: {n} x {m} | {off} | {ok} | {missing} |")
        break

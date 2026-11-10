import ctypes, sys, numpy as np
ROOT="/nvme0n1-disk/code/mojo-jiwer"; sys.path.insert(0, ROOT+"/python")
from mojo_jiwer import _native
I64=ctypes.c_int64; A=lambda x:int(x.ctypes.data)
lib=ctypes.CDLL(ROOT+"/dist/libmojo-jiwer.so")
lib.mji_trace.argtypes=[I64]*6
BAD=7
def sentinel(m):
    # matrix is (n+1)*(m+1); base = alloc aligned to >=16, start at buffer+off
    cells=(m+1)*(m+1)+64
    buf=np.full(cells, BAD, dtype=np.int32)
    # choose an offset so the matrix start is 4-byte aligned but not 16
    off={0:0,1:1,2:2,3:3,4:4}[ (4-m) % 4 ] if False else None
    return buf
for m in (0,1,7,8,9,15,16,17,20,63):
    for start_off in range(0,16):
        cells=(m+1)*(m+1)+start_off+16
        buf=np.full(cells-16, BAD, dtype=np.int32)
        view=buf[start_off:start_off+(m+1)*(m+1)]
        ref=np.zeros(0,dtype=np.int64)      # n=0 stops the DP loop; still inits boundary
        hyp=np.arange(m,dtype=np.int64)
        ops=np.zeros(max(1,m),dtype=np.uint8)
        # n comes from reference_len: use 1 so the DP loop runs once and the
        # boundary is read as part of row 1
        ref=np.zeros(1,dtype=np.int64)
        cnt=lib.mji_trace(A(ref),A(hyp),1,m,A(view),A(ops))
        row0=view[:m+1].copy()
        if row0.tolist()!=list(range(m+1)):
            print(f"m={m} start_off={start_off}: row0={row0[:12].tolist()}")

import ctypes, sys
ROOT="/nvme0n1-disk/code/mojo-jiwer"
sys.path.insert(0, ROOT+"/python")
import numpy as np
from mojo_jiwer import _native
I64=ctypes.c_int64; A=_native._address
pattern=list(range(1,66)); text=pattern+[1]
ref=np.array(pattern,dtype=np.int64); hyp=np.array(text,dtype=np.int64)
ro=np.array([0,65],dtype=np.int64); ho=np.array([0,66],dtype=np.int64)
mws=1; mso=66
masks=np.zeros(mso+2*mws,dtype=np.uint64); out=np.empty(1,dtype=np.int64)
lib=ctypes.CDLL(sys.argv[1]); lib.mji_distances.argtypes=[I64]*12
lib.mji_distances(A(ref),A(hyp),A(ro),A(ho),1,1,0,A(masks),0,mws,mso,A(out))
# which hint slots differ from mid-run state? reproduce run by hand is complex;
# instead: after the call, is slot[0] (token 1) dirty? A correct implementation
# leaves it dirty (mask reuse is per call); check that it is exactly 1<<0.
print("slot(token1) =",masks[0], "dist",int(out[0]))

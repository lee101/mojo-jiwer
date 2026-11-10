import ctypes, numpy as np
k2=ctypes.CDLL('/nvme0n1-disk/code/mojo-jiwer/.parwork/kernel2.so')
k2.k_serial.argtypes=[ctypes.c_int64]*10; k2.k_serial.restype=ctypes.c_int64
R=[[i%7 for i in range(L)] for L in (27,2,17,18)]
H=[list(r) for r in R]
n=len(R)
ro=np.zeros(n+1,dtype=np.int64); ho=np.zeros(n+1,dtype=np.int64)
ro[1:]=np.cumsum([len(r) for r in R]); ho[1:]=np.cumsum([len(h) for h in H])
ref=np.array([t for r in R for t in r],dtype=np.int64); hyp=np.array([t for h in H for t in h],dtype=np.int64)
for i in range(len(ref)): ref[i]=i%5
for i in range(len(hyp)): hyp[i]=i%5
mws=1; mso=30; stride=32
masks=np.zeros(stride*2,dtype=np.uint64); res=np.zeros(n,dtype=np.int64)
print("ref",hex(ref.ctypes.data),"hyp",hex(hyp.ctypes.data),"masks",hex(masks.ctypes.data),"res",hex(res.ctypes.data),flush=True)
ref0=ref.copy(); hyp0=hyp.copy()
k2.k_serial(ref.ctypes.data,hyp.ctypes.data,ro.ctypes.data,ho.ctypes.data,res.ctypes.data,masks.ctypes.data,stride,mso,n,1)
print("res",res,"ref changed",(ref!=ref0).sum(),"hyp changed",(hyp!=hyp0).sum(),flush=True)

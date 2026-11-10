import ctypes, numpy as np, random, sys
base=ctypes.CDLL('/nvme0n1-disk/code/mojo-jiwer/.parwork/baseline.so')
k2=ctypes.CDLL('/nvme0n1-disk/code/mojo-jiwer/.parwork/kernel2.so')
base.mji_distances.argtypes=[ctypes.c_int64]*12
k2.k_serial.argtypes=[ctypes.c_int64]*10; k2.k_serial.restype=ctypes.c_int64
rng=random.Random(1); n=50
R=[];H=[]
for _ in range(n):
    L=rng.randint(1,20); r=[rng.randrange(30) for _ in range(L)]; h=r.copy()
    for i in range(0,L,9): h[i]=rng.randrange(30)
    R.append(r); H.append(h)
ro=np.zeros(n+1,dtype=np.int64); ho=np.zeros(n+1,dtype=np.int64)
ro[1:]=np.cumsum([len(r) for r in R]); ho[1:]=np.cumsum([len(h) for h in H])
ref=np.array([t for r in R for t in r],dtype=np.int64); hyp=np.array([t for h in H for t in h],dtype=np.int64)
mws=max(1,(int(np.diff(ho).max())+63)//64); mso=(max(int(ref.max()),int(hyp.max()))+1)*mws
stride=mso+2*mws
print("mws",mws,"mso",mso,"stride",stride)
masks=np.zeros(stride,dtype=np.uint64); out=np.zeros(n,dtype=np.int64)
# baseline expects max(ref,hyp) length ordering per pair; feed as-is
base.mji_distances(ref.ctypes.data,hyp.ctypes.data,ro.ctypes.data,ho.ctypes.data,n,1,0,masks.ctypes.data,0,mws,mso,out.ctypes.data)
print("baseline", out[:8])
ref0=ref.copy(); hyp0=hyp.copy()
ref[:]=-1; hyp[:]=-1
res=np.zeros(n,dtype=np.int64)
r=k2.k_serial(ref.ctypes.data,hyp.ctypes.data,ro.ctypes.data,ho.ctypes.data,res.ctypes.data,masks.ctypes.data,mso,stride,n,1)
print("ret",r,"kernel2", res[:8])
print("overwritten ref/hyp:", (ref!=ref0).sum(), (hyp!=hyp0).sum())
ref, hyp = ref0, hyp0
print("equal", (res==out).all())

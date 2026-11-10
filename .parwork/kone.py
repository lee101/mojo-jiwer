import ctypes, numpy as np, sys, random
lib=ctypes.CDLL('/nvme0n1-disk/code/mojo-jiwer/.parwork/kernel.so')
for name in ('k_parallel','k_serial'):
    f=getattr(lib,name); f.argtypes=[ctypes.c_int64]*10; f.restype=ctypes.c_int64
rng=random.Random(7); vocab=list(range(256))
n=int(sys.argv[1]); w=int(sys.argv[2]); fn=sys.argv[3]
R=[];H=[]
for _ in range(n):
    L=rng.randint(6,40); r=[rng.choice(vocab) for _ in range(L)]; h=r.copy()
    for i in range(0,L,9): h[i]=rng.choice(vocab)
    R.append(r); H.append(h)
ro=np.zeros(n+1,dtype=np.int64); ho=np.zeros(n+1,dtype=np.int64)
np.cumsum([len(r) for r in R],out=ro[1:]); np.cumsum([len(h) for h in H],out=ho[1:])
ref=np.array([t for r in R for t in r],dtype=np.int64); hyp=np.array([t for h in H for t in h],dtype=np.int64)
stride=256
res=np.zeros(n,dtype=np.int64); masks=np.zeros(max(1,w*stride*4+4096),dtype=np.uint64)
print(getattr(lib,fn)(ref.ctypes.data,hyp.ctypes.data,ro.ctypes.data,ho.ctypes.data,res.ctypes.data,masks.ctypes.data,stride,n,0,w), flush=True)
print(res[:5], flush=True)
import os; os._exit(0)

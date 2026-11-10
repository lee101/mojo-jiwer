import ctypes, numpy as np, random
k2=ctypes.CDLL('/nvme0n1-disk/code/mojo-jiwer/.parwork/kernel2.so')
k2.k_serial.argtypes=[ctypes.c_int64]*10; k2.k_serial.restype=ctypes.c_int64
def run(R,H):
    n=len(R)
    ro=np.zeros(n+1,dtype=np.int64); ho=np.zeros(n+1,dtype=np.int64)
    ro[1:]=np.cumsum([len(r) for r in R]); ho[1:]=np.cumsum([len(h) for h in H])
    ref=np.array([t for r in R for t in r],dtype=np.int64); hyp=np.array([t for h in H for t in h],dtype=np.int64)
    mws=max(1,(int(np.diff(ho).max())+63)//64); mso=(max(int(ref.max()),int(hyp.max()))+1)*mws
    stride=mso+2*mws
    masks=np.zeros(stride*3,dtype=np.uint64); res=np.zeros(n,dtype=np.int64)
    ref0=ref.copy(); hyp0=hyp.copy()
    k2.k_serial(ref.ctypes.data,hyp.ctypes.data,ro.ctypes.data,ho.ctypes.data,res.ctypes.data,masks.ctypes.data,stride,mso,n,1)
    return res, (ref!=ref0).sum()+(hyp!=hyp0).sum(), masks
rng=random.Random(0)
for trial,(lo,hi,voc) in enumerate([(1,5,4),(1,20,30),(1,64,4),(1,64,30),(1,70,30)]):
    for rep in range(3):
        R=[rng.choices(range(voc),k=rng.randint(lo,hi)) for _ in range(5)]
        H=[list(r) for r in R]
        for r,h in zip(R,H):
            if r: h[rng.randrange(len(r))]=rng.randrange(voc)
        res,changed,mk=run(R,H)
        print(trial,lo,hi,voc,"changed",changed,"dirty_mask_words",int((mk!=0).sum()),"res",res)

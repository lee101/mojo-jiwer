import ctypes, time, statistics, random, numpy as np, sys
lib=ctypes.CDLL('/nvme0n1-disk/code/mojo-jiwer/.parwork/kernel.so')
for name in ('k_par','k_serial'):
    f=getattr(lib,name); f.argtypes=[ctypes.c_int64]*10; f.restype=ctypes.c_int64
rng=random.Random(7)
vocab=list(range(256))
def corpus(count, lo, hi):
    R=[];H=[]
    for _ in range(count):
        L=rng.randint(lo,hi)
        r=[rng.choice(vocab) for _ in range(L)]
        h=r.copy()
        for i in range(0,L,9): h[i]=rng.choice(vocab)
        R.append(r); H.append(h)
    return R,H
def prep(R,H):
    ro=np.zeros(len(R)+1,dtype=np.int64); ho=np.zeros(len(H)+1,dtype=np.int64)
    np.cumsum([len(r) for r in R],out=ro[1:]); np.cumsum([len(h) for h in H],out=ho[1:])
    ref=np.array([t for r in R for t in r] or [0],dtype=np.int64)
    hyp=np.array([t for h in H for t in h] or [0],dtype=np.int64)
    rl=np.diff(ro); hl=np.diff(ho)
    stride=int(max(rl.max(),hl.max()))+1
    return ref,hyp,ro,ho,stride
def run(key, count, lo, hi, workers, fn='k_serial'):
    R,H=key[:2]
    ref,hyp,ro,ho,stride=key[2:]
    res=np.zeros(count,dtype=np.int64)
    masks=np.zeros(max(1,workers*stride),dtype=np.uint64)
    f=getattr(lib,fn)
    args=(ref.ctypes.data,hyp.ctypes.data,ro.ctypes.data,ho.ctypes.data,res.ctypes.data,masks.ctypes.data,stride,count)
    t=time.process_time()
    if workers==1:
        f(*args,0,1)
    else:
        # use max.algorithm.parallelize inside k_par; call chunk 0 with workers
        f(*args,0,workers)
    return time.process_time()-t, res
for label,count,lo,hi in [("2k x 6-40w",2000,6,40),("20k x 6-40w",20000,6,40),("5k x 20w",5000,20,20),("50k x 6-40w",50000,6,40)]:
    R,H=corpus(count,lo,hi); key=(R,H)+prep(R,H)
    base,ref_res=run(key,count,lo,hi,1)
    print(label,"serial",f"{base*1e3:.2f} ms cpu",flush=True)
    for w in (2,4,8):
        for _ in range(3):
            t,res=run(key,count,lo,hi,w,fn='k_par')
        assert (res==ref_res).all(), "mismatch"
        print(label,f"k_par w{w}",f"{t*1e3:.2f} ms cpu",f"{base/t:.2f}x",flush=True)

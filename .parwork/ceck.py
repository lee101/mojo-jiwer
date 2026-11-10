import numpy as np, xover, os, random
lib = xover.load('baseline.so')
def char_corpus(count,length,seed):
    rng=random.Random(seed); alpha="abcdefghijklmnopqrstuvwxyz "
    R=[];H=[]
    for _ in range(count):
        r=[rng.choice(alpha) for _ in range(length)]
        h=r.copy()
        for i in range(0,length,11): h[i]=rng.choice(alpha)
        R.append([ord(c) for c in r]); H.append([ord(c) for c in h])
    return R,H
R,H=char_corpus(5000,48,3)
for w in (1,1,2,4,1,2,4,1):
    go,out=xover.dist_blob(lib,R,H,5000,w)
    print(w, f"{xover.best(go,out)*1e3:.2f} ms")

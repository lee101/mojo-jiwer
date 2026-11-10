import xover, os, random, numpy as np
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
cases=[]
for c,l in [(1000,12),(2000,12),(5000,12),(10000,12),(20000,12),(2000,20),(5000,20),(2000,60),(1000,60)]:
    R,H=xover.word_corpus(c,l,c+7); cases.append((f"dist {c}x{l}w","dist",R,H))
for c,l in [(2000,20),(5000,20),(10000,20),(2000,12),(10000,12),(1000,60),(2000,60)]:
    R,H=xover.word_corpus(c,l,c+7); cases.append((f"trace {c}x{l}w","trace",R,H))
R,H=char_corpus(5000,48,3); cases.append(("dist 5000x48c","dist",R,H))
R,H=char_corpus(20000,48,3); cases.append(("dist 20000x48c","dist",R,H))
ws=[1,2,3,4,8]
print("| case | " + " | ".join(f"w={w}" for w in ws) + " |")
print("|---|" + "---:|"*len(ws))
for name,kind,R,H in cases:
    count=len(R); row=[]; base=None
    for w in ws:
        b = xover.dist_blob if kind=="dist" else xover.trace_blob
        go,out=b(lib,R,H,count,w)
        t=xover.best(go,out,5)
        if base is None: base=t
        row.append(f"{t*1e3:.2f} ({base/t:.2f}x)")
    print(f"| {name} | "+" | ".join(row)+" |")

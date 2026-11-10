import sys, resource
ROOT="/nvme0n1-disk/code/mojo-jiwer"; sys.path.insert(0, ROOT+"/python"); sys.path.insert(0, ROOT+"/bench")
import bench as b, mojo_jiwer as mojo
from mojo_jiwer import _native
from mojo_jiwer.process import _default_transform, _encode, _chunks, AlignmentChunk
def cpu():
    r=resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime+r.ru_stime
def best(f,reps=5):
    t=float("inf")
    for _ in range(reps):
        c=cpu(); f(); t=min(t,cpu()-c)
    return t
r4,h4=b.word_corpus(2_000,20,4)
rt=_default_transform(r4,__import__("mojo_jiwer.transformations",fromlist=["x"]).wer_default,True)
ht=_default_transform(h4,__import__("mojo_jiwer.transformations",fromlist=["x"]).wer_default,False)
a=_encode(rt,ht)
tr=_native.traces(a[0],a[1])
names=("equal","substitute","delete","insert")
def counts_only():
    hits=subs=dels=ins=0
    keep=AlignmentChunk
    for t in tr:
        n=len(t); s=0
        while s<n:
            op=t[s]; ref=s; hyp=s
            e=s
            while e<n and t[e]==op: e+=1
            d=e-s
            if op==0: hits+=d; ref+=d; hyp+=d
            elif op==1: subs+=d; ref+=d; hyp+=d
            elif op==2: dels+=d; ref+=d
            else: ins+=d; hyp+=d
            s=e
    return hits,subs,dels,ins
full=best(lambda: [_chunks(t) for t in tr])
cnt=best(counts_only)
print(f"- _chunks (build chunk objects): {full*1e3:.3f} ms")
print(f"- same decomposition, counts only  : {cnt*1e3:.3f} ms")
print(f"- per-sentence list construction + arithmetic only: {(full-cnt)*1e3:.3f} ms")

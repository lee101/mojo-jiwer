import sys, resource
ROOT="/nvme0n1-disk/code/mojo-jiwer"; sys.path.insert(0, ROOT+"/python"); sys.path.insert(0, ROOT+"/bench")
import bench as b, mojo_jiwer as mojo, jiwer as upstream
from mojo_jiwer import _native
from mojo_jiwer.process import _default_transform, _encode, _chunks, AlignmentChunk
from mojo_jiwer.transformations import wer_default
def cpu():
    r=resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime+r.ru_stime
def best(f,reps=7):
    t=float("inf")
    for _ in range(reps):
        c=cpu(); f(); t=min(t,cpu()-c)
    return t
r4,h4=b.word_corpus(2_000,20,4)
rt=_default_transform(r4,wer_default,True); ht=_default_transform(h4,wer_default,False); a=_encode(rt,ht)
tr=_native.traces(a[0],a[1])
def counts_only():
    for t in tr:
        n=len(t); s=0
        while s<n:
            op=t[s]; e=s
            while e<n and t[e]==op: e+=1
            s=e
def loop_with_chunks():
    for rr,hh,ops in zip(a[0],a[1],tr):
        cs=_chunks(ops)
        for c in cs:
            if c.type=="equal": pass
print(f"_chunks only                    : {best(lambda: [_chunks(t) for t in tr])*1e3:.3f} ms")
print(f"scan-only (no chunk objects)     : {best(counts_only)*1e3:.3f} ms")
print(f"public process_words (2k x 20w)  : {best(lambda: mojo.process_words(r4,h4))*1e3:.2f} ms")
print(f"public upstream  (2k x 20w)      : {best(lambda: upstream.process_words(r4,h4))*1e3:.2f} ms")

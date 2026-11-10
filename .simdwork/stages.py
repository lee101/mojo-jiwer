import sys, resource
ROOT="/nvme0n1-disk/code/mojo-jiwer"; sys.path.insert(0, ROOT+"/python"); sys.path.insert(0, ROOT+"/bench")
import numpy as np, bench as b, mojo_jiwer as mojo
from mojo_jiwer import _native
from mojo_jiwer.process import _default_transform, _encode, _chunks
from mojo_jiwer.transformations import wer_default, cer_default
def cpu():
    r=resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime+r.ru_stime
def best(f,reps=5):
    t=float("inf")
    for _ in range(reps):
        c=cpu(); f(); t=min(t,cpu()-c)
    return t
def stages(label,refs,hyps,kind,public):
    tf=wer_default if kind!="cer" else cer_default
    rt=_default_transform(refs,tf,True); ht=_default_transform(hyps,tf,False); a=_encode(rt,ht)
    print(f"### {label}")
    print(f"- PUBLIC: {best(public)*1e3:.3f} ms")
    print(f"- default_transform (both sides): {best(lambda: [_default_transform(refs,tf,True),_default_transform(hyps,tf,False)])*1e3:.3f} ms")
    print(f"- encode: {best(lambda: _encode(rt,ht))*1e3:.3f} ms")
    if kind=="cer":
        ro=np.empty(len(refs)+1,dtype=np.int64); ho=np.empty(len(hyps)+1,dtype=np.int64); ro[0]=ho[0]=0
        rl=[len(v.strip()) for v in refs]; hl=[len(v.strip()) for v in hyps]
        np.cumsum(np.fromiter(rl,dtype=np.int64,count=len(rl)),out=ro[1:])
        np.cumsum(np.fromiter(hl,dtype=np.int64,count=len(hl)),out=ho[1:])
        r=np.frombuffer("".join(v.strip() for v in refs).encode("utf-32le"),dtype="<u4").astype(np.int64)
        h=np.frombuffer("".join(v.strip() for v in hyps).encode("utf-32le"),dtype="<u4").astype(np.int64)
        print(f"- kernel mji_distances (whole CER batch): {best(lambda: _native.distances_flat(r,h,ro,ho))*1e3:.3f} ms")
    elif kind=="words":
        print(f"- _native.traces: {best(lambda: _native.traces(a[0],a[1]),3)*1e3:.3f} ms")
        tr=_native.traces(a[0],a[1])
        print(f"- _chunks over all traces: {best(lambda: [_chunks(t) for t in tr],3)*1e3:.3f} ms")
    else:
        print(f"- _native.distances (whole batch): {best(lambda: _native.distances(a[0],a[1]))*1e3:.3f} ms")
    print()
r1,h1=b.word_corpus(10_000,12,1); stages("WER 10k x 12w",r1,h1,"wer",lambda: mojo.wer(r1,h1))
r2,h2=b.word_corpus(2_000,60,2); stages("WER 2k x 60w",r2,h2,"wer",lambda: mojo.wer(r2,h2))
r3,h3=b.character_corpus(5_000,48,3); stages("CER 5k x 48c",r3,h3,"cer",lambda: mojo.cer(r3,h3))
r4,h4=b.word_corpus(2_000,20,4); stages("process_words 2k x 20w",r4,h4,"words",lambda: mojo.process_words(r4,h4))

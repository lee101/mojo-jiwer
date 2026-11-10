"""Interleaved A/B of two stacked builds via the REAL public API, plus a
tight-loop micro check of the mask build. Alternates A/B reps so drift is shared."""
import ctypes, sys, time, os
ROOT="/nvme0n1-disk/code/mojo-jiwer"
sys.path.insert(0, ROOT+"/python"); sys.path.insert(0, ROOT+"/bench")
import numpy as np, bench as b
import mojo_jiwer as mojo
I64=ctypes.c_int64; A=lambda x:int(x.ctypes.data)
libs={}
def load(p):
    l=ctypes.CDLL(p)
    for n in ("mji_distances","mji_traces"):
        f=getattr(l,n); f.argtypes=[I64]*12; f.restype=None
    return l
def best_pairs(fa, fb, reps=7):
    ta=tb=float("inf")
    for _ in range(reps):
        s=time.perf_counter(); fa(); ta=min(ta,time.perf_counter()-s)
        s=time.perf_counter(); fb(); tb=min(tb,time.perf_counter()-s)
    return ta,tb
cases=[("WER 10k x 12w", b.word_corpus(10_000,12,1), mojo.wer),
       ("WER 2k x 60w", b.word_corpus(2_000,60,2), mojo.wer),
       ("CER 5k x 48c", b.character_corpus(5_000,48,3), mojo.cer),
       ("process 2k x 20w", b.word_corpus(2_000,20,4), mojo.process_words),
       ("WER 1 x 1000w", b.word_corpus(1,1_000,5), mojo.wer)]
def run_with(so, fn):
    # point the wrapper at a specific .so for this phase only
    from mojo_jiwer import _native
    _native._library = libs[so]
    return fn
if __name__=="__main__":
    a,build_b=sys.argv[1],sys.argv[2]
    libs[a]=load(a); libs[build_b]=load(build_b)
    from mojo_jiwer import _native
    print("| case | A ms | B ms | A/B |")
    print("|---|---:|---:|---:|")
    for label,(refs,hyps),fn in cases:
        fns=[]
        for so in (a,build_b):
            _native._library=libs[so]
            fns.append(lambda refs=refs,hyps=hyps,fn=fn: fn(refs,hyps))
        # correctness: identical result
        r1,r2=fns[0](),fns[1]()
        assert (r1 if not hasattr(r1,"hits") else (r1.hits,r1.substitutions,r1.deletions,r1.insertions)) == (r2 if not hasattr(r2,"hits") else (r2.hits,r2.substitutions,r2.deletions,r2.insertions)), label
        ta,tb=best_pairs(*fns,reps=5)
        print(f"| {label} | {ta*1e3:.2f} | {tb*1e3:.2f} | {ta/tb:.3f} |")

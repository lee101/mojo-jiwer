import sys, time, resource, statistics
ROOT="/nvme0n1-disk/code/mojo-jiwer"; sys.path.insert(0, ROOT+"/python"); sys.path.insert(0, ROOT+"/bench")
import bench as b, mojo_jiwer as mojo, jiwer as upstream
from mojo_jiwer.process import _chunks, WordOutput, AlignmentChunk
from collections import defaultdict
def cpu():
    r=resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime+r.ru_stime
names=("equal","substitute","delete","insert")
def fused(reference,hypothesis):
    # same normalisation + encoding as process_words
    from mojo_jiwer.process import _prepare
    from mojo_jiwer import _native
    references,hypotheses,ref_ids,hyp_ids=_prepare(reference,hypothesis,
        __import__("mojo_jiwer.transformations",fromlist=["x"]).wer_default,
        __import__("mojo_jiwer.transformations",fromlist=["x"]).wer_default)
    alignments=[]; hits=subs=dels=ins=0; rw=hw=0
    for ref,hyp,ops in zip(ref_ids,hyp_ids,_native.traces(ref_ids,hyp_ids)):
        chunks=[]; append=chunks.append; n=len(ops); s=0
        ri=hi=0
        while s<n:
            op=ops[s]; rs=ri; hs=hi; e=s
            while e<n and ops[e]==op:
                if op!=3: ri+=1
                if op!=2: hi+=1
                e+=1
            d=e-s
            if op==0: hits+=d
            elif op==1: subs+=d
            elif op==2: dels+=d
            else: ins+=d
            append(AlignmentChunk(names[op],rs,ri,hs,hi))
            s=e
        alignments.append(chunks); rw+=len(ref); hw+=len(hyp)
    errors=subs+dels+ins
    return WordOutput(references,hypotheses,alignments,errors/(hits+subs+dels),
                      errors/(hits+subs+dels+ins),0.0,0.0,hits,subs,ins,dels)
r4,h4=b.word_corpus(2_000,20,4)
def sig(o): return (o.hits,o.substitutions,o.deletions,o.insertions,o.wer,o.mer,
    [(c.type,c.ref_start_idx,c.ref_end_idx,c.hyp_start_idx,c.hyp_end_idx) for cs in o.alignments for c in cs])
print("fused == current:", sig(fused(r4,h4))==sig(mojo.process_words(r4,h4)))
ta=[];tb=[]
for _ in range(11):
    c=time.perf_counter(); mojo.process_words(r4,h4); ta.append(time.perf_counter()-c)
    c=time.perf_counter(); fused(r4,h4); tb.append(time.perf_counter()-c)
print(f"current  median {statistics.median(ta)*1e3:.2f} ms   min {min(ta)*1e3:.2f}")
print(f"fused    median {statistics.median(tb)*1e3:.2f} ms   min {min(tb)*1e3:.2f}")
ca=[];cb=[]
for _ in range(15):
    c=cpu(); mojo.process_words(r4,h4); ca.append(cpu()-c)
    c=cpu(); fused(r4,h4); cb.append(cpu()-c)
print(f"CPU min: current {min(ca)*1e3:.2f} ms, fused {min(cb)*1e3:.2f} ms")

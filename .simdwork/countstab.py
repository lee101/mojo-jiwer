import sys, time, resource, statistics
ROOT="/nvme0n1-disk/code/mojo-jiwer"; sys.path.insert(0, ROOT+"/python"); sys.path.insert(0, ROOT+"/bench")
import bench as b, jiwer as upstream, mojo_jiwer as mojo
from mojo_jiwer.process import _chunks
from collections import Counter
def cpu():
    r=resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime+r.ru_stime
r4,h4=b.word_corpus(2_000,20,4)
# is _chunks deterministic + are inputs identical run to run?
a=mojo.process_words(r4,h4); bb=mojo.process_words(r4,h4)
sig=lambda o:(o.hits,o.substitutions,o.deletions,o.insertions,[(c.type,c.ref_start_idx,c.ref_end_idx,c.hyp_start_idx,c.hyp_end_idx) for cs in o.alignments for c in cs])
print("process_words deterministic:", sig(a)==sig(bb))
ta=[];tb=[]
for _ in range(9):
    c=time.perf_counter(); mojo.process_words(r4,h4); ta.append(time.perf_counter()-c)
    c=time.perf_counter(); upstream.process_words(r4,h4); tb.append(time.perf_counter()-c)
print(f"process_words ours  median {statistics.median(ta)*1e3:.2f} ms  min {min(ta)*1e3:.2f}")
print(f"process_words upst. median {statistics.median(tb)*1e3:.2f} ms  min {min(tb)*1e3:.2f}")
ca=[];cb=[]
for _ in range(15):
    c=cpu(); mojo.process_words(r4,h4); ca.append(cpu()-c)
    c=cpu(); upstream.process_words(r4,h4); cb.append(cpu()-c)
print(f"CPU min: ours {min(ca)*1e3:.2f} ms, upstream {min(cb)*1e3:.2f} ms")

import sys, resource
ROOT="/nvme0n1-disk/code/mojo-jiwer"; sys.path.insert(0, ROOT+"/python"); sys.path.insert(0, ROOT+"/bench")
import bench as b, mojo_jiwer as mojo
def cpu():
    r=resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime+r.ru_stime
r4,h4=b.word_corpus(2_000,20,4)
best=float("inf")
for _ in range(7):
    c=cpu(); mojo.process_words(r4,h4); best=min(best,cpu()-c)
print(f"process_words 2k x 20w public, min of 7 CPU-time calls: {best*1e3:.2f} ms")

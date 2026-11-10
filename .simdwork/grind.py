import sys, resource
ROOT="/nvme0n1-disk/code/mojo-jiwer"; sys.path.insert(0, ROOT+"/python"); sys.path.insert(0, ROOT+"/bench")
import bench as b, mojo_jiwer as mojo
def cpu():
    r=resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime+r.ru_stime
r4,h4=b.word_corpus(2_000,20,4)
best=float("inf")
for _ in range(15):
    c=cpu()
    for _ in range(3): mojo.process_words(r4,h4)
    best=min(best,(cpu()-c)/3)
print(f"process_words 2k x 20w, 3 calls per CPU-time sample, min of 15: {best*1e3:.2f} ms")

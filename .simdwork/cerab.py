import sys, resource
ROOT="/nvme0n1-disk/code/mojo-jiwer"; sys.path.insert(0, ROOT+"/python"); sys.path.insert(0, ROOT+"/bench")
import bench as b, jiwer as upstream, mojo_jiwer as mojo
def cpu():
    r=resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime+r.ru_stime
refs,hyps=b.character_corpus(5_000,48,3)
assert mojo.cer(refs,hyps)==upstream.cer(refs,hyps)
ta=tb=float("inf")
for _ in range(40):
    c=cpu(); mojo.cer(refs,hyps); ta=min(ta,cpu()-c)
    c=cpu(); upstream.cer(refs,hyps); tb=min(tb,cpu()-c)
print(f"CER 5k x 48c (5 inner reps): mojo {ta*1e3:.3f} ms cpu, upstream {tb*1e3:.3f} ms cpu")

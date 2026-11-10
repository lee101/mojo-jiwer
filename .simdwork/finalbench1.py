import sys, resource
ROOT="/nvme0n1-disk/code/mojo-jiwer"; sys.path.insert(0, ROOT+"/python")
sys.path.insert(0, ROOT+"/bench")
import bench as b, jiwer as upstream, mojo_jiwer as mojo
def cpu():
    r=resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime+r.ru_stime
CASES=[("WER: 10,000 x 12-word utterances", b.word_corpus(10_000,12,1), "wer"),
       ("WER: 2,000 x 60-word utterances", b.word_corpus(2_000,60,2), "wer"),
       ("CER: 5,000 x 48-character utterances", b.character_corpus(5_000,48,3), "cer"),
       ("process_words: 2,000 x 20 words", b.word_corpus(2_000,20,4), "process_words"),
       ("WER: one 1,000-word transcript", b.word_corpus(1,1_000,5), "wer")]
print("| Case | mojo-jiwer (min cpu) | jiwer (min cpu) |")
print("|---|---:|---:|")
for label,(r,h),kind in CASES:
    fn=getattr(mojo,kind); up=getattr(upstream,kind)
    tm=tu=float("inf")
    for _ in range(5):
        c=cpu(); fn(r,h); tm=min(tm,cpu()-c)
        c=cpu(); up(r,h); tu=min(tu,cpu()-c)
    print(f"| {label} | {tm*1e3:.2f} ms | {tu*1e3:.2f} ms |")

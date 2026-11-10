import xover, os, time, numpy as np, statistics
lib=xover.load('baseline.so')
R,H=xover.word_corpus(5000,20,5009)
for w in (1,2,4,1,4,2,1):
    go,out=xover.dist_blob(lib,R,H,5000,w)
    go()
    ts=[]
    for _ in range(9):
        t=time.process_time(); go(); ts.append(time.process_time()-t)
    print(w, f"cpu {statistics.median(ts)*1e3:.2f} ms")
# also wall
for w in (1,2,4):
    go,out=xover.dist_blob(lib,R,H,5000,w); go()
    ts=[]
    for _ in range(9):
        t=time.perf_counter(); go(); ts.append(time.perf_counter()-t)
    print("wall",w,f"{statistics.median(ts)*1e3:.2f} ms")

import ctypes, time, statistics, numpy as np, os
lib=ctypes.CDLL('/nvme0n1-disk/code/mojo-jiwer/.parwork/probe.so')
lib.probe_parallel_call.argtypes=[ctypes.c_int64]*3; lib.probe_parallel_call.restype=ctypes.c_int64
n=20_000_000
buf=np.zeros(n,dtype=np.int64)
for w in (1,2,3,4,8):
    ws=[];cs=[]
    for _ in range(7):
        t=time.perf_counter(); c=time.process_time()
        lib.probe_parallel_call(buf.ctypes.data,n,w)
        ws.append(time.perf_counter()-t); cs.append(time.process_time()-c)
    print(w, f"wall {statistics.median(ws)*1e3:.2f} ms cpu {statistics.median(cs)*1e3:.2f} ms", flush=True)
print("affinity", len(os.sched_getaffinity(0)))

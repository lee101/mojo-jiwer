import ctypes, time, statistics, numpy as np
lib=ctypes.CDLL('/nvme0n1-disk/code/mojo-jiwer/.parwork/probe.so')
lib.probe_parallel_call.argtypes=[ctypes.c_int64]*3; lib.probe_parallel_call.restype=ctypes.c_int64
def cpu(): return time.process_time()
n=1000000
buf=np.zeros(n,dtype=np.int64)
for w in (1,2,4,8):
    ts=[]
    for _ in range(9):
        t=cpu(); lib.probe_parallel_call(buf.ctypes.data,n,w); ts.append(cpu()-t)
    print("max.parallelize",w,f"cpu {statistics.median(ts)*1e3:.2f} ms", flush=True)

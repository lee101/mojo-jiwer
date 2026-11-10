import ctypes, time, statistics
lib=ctypes.CDLL('/nvme0n1-disk/code/mojo-jiwer/.parwork/probe.so')
lib.probe_parallel_call.argtypes=[ctypes.c_int64]*3; lib.probe_parallel_call.restype=ctypes.c_int64
buf=__import__('numpy').zeros(1,dtype=__import__('numpy').int64)
# measure launch overhead: workers = n work items, empty body (worker writes nothing when n==0? body still runs per item)
for w in (1,2,4,8,16):
    ws=[]
    for _ in range(11):
        t=time.perf_counter(); lib.probe_parallel_call(buf.ctypes.data,0,w); ws.append(time.perf_counter()-t)
    print(w, f"{statistics.median(ws)*1e3:.3f} ms", flush=True)

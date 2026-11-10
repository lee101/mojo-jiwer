import ctypes, time, statistics, numpy as np
from concurrent.futures import ThreadPoolExecutor
lib=ctypes.CDLL('probe.so')
lib.probe_parallel_call.argtypes=[ctypes.c_int64]*3; lib.probe_parallel_call.restype=ctypes.c_int64
def spin(us):
    t=time.perf_counter_ns()
    while time.perf_counter_ns()-t < us*1000: pass
def cpu():
    return time.process_time()
buf=np.zeros(1,dtype=np.int64)
for w in (2,4,8):
    ts=[]
    for _ in range(9):
        t=cpu(); lib.probe_parallel_call(buf.ctypes.data,1000000,w); ts.append(cpu()-t)
    print("max.parallelize",w,f"{statistics.median(ts)*1e3:.2f} ms")
for w in (2,4,8):
    ts=[]
    for _ in range(9):
        t=cpu()
        with ThreadPoolExecutor(max_workers=w) as ex: list(ex.map(spin,[3000]*w))
        ts.append(cpu()-t)
    print("py threads spin",w,f"{statistics.median(ts)*1e3:.2f} ms")

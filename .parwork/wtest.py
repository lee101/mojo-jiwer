import ctypes, numpy as np
lib=ctypes.CDLL('/nvme0n1-disk/code/mojo-jiwer/.parwork/workers.so')
lib.count_workers.argtypes=[ctypes.c_int64]*4; lib.count_workers.restype=ctypes.c_int64
lib.spin_ns.argtypes=[ctypes.c_int64]; lib.spin_ns.restype=ctypes.c_int64
slots=np.zeros(1000,dtype=np.int64)
for workers in (2,4):
    slots[:]=0
    lib.count_workers(slots.ctypes.data, 400, workers, 0)
    t=slots[slots>0]
    t.sort()
    print("workers",workers,"items recorded",len(t),"distinct start times",len(np.unique(t)),flush=True)

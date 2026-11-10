import ctypes
import os
import statistics
import time

import numpy as np

lib = ctypes.CDLL("/nvme0n1-disk/code/mojo-jiwer/.parwork/items.so")
lib.item_work.argtypes = [ctypes.c_int64] * 3
lib.item_work.restype = ctypes.c_int64

buf = np.zeros(1024 * 8, dtype=np.int64)
for workers in (1, 2, 3, 4):
    wall, cpu = [], []
    for _ in range(11):
        t = time.perf_counter()
        c = time.process_time()
        lib.item_work(buf.ctypes.data, 800, workers)
        wall.append(time.perf_counter() - t)
        cpu.append(time.process_time() - c)
    print(f"workers={workers}: wall {statistics.median(wall)*1e3:.2f} ms "
          f"cpu {statistics.median(cpu)*1e3:.2f} ms", flush=True)

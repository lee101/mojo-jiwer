import ctypes
import statistics

import numpy as np

lib = ctypes.CDLL("/nvme0n1-disk/code/mojo-jiwer/.parwork/items2.so")
lib.item_spin.argtypes = [ctypes.c_int64] * 4
lib.item_spin.restype = ctypes.c_int64

n = 400
spin = 50_000  # ns per item
for workers in (1, 2, 4):
    for _ in range(3):
        slots = np.zeros(2 * n, dtype=np.int64)
        lib.item_spin(slots.ctypes.data, n, workers, spin)
        start = slots[0::2]
        end = slots[1::2]
        # merge overlapping intervals to estimate concurrent threads
        order = np.argsort(start)
        end_sorted = end[order]
        running_end = 0
        lanes = 0
        for e in end_sorted:
            if start[order][0] == 0:
                pass
            if e > running_end:
                lanes += 1
                running_end = e
        # simpler: total busy time / wall span
        span = int(end.max() - start.min())
        busy = int((end - start).sum())
        print(f"workers={workers} span={span/1e6:.2f}ms busy={busy/1e6:.2f}ms "
              f"busy/span={busy/max(span,1):.2f}", flush=True)

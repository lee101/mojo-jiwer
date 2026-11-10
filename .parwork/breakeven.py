"""Launch cost vs. speedup: where does in-Mojo parallelism break even?"""

import ctypes
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

lib = ctypes.CDLL("/nvme0n1-disk/code/mojo-jiwer/.parwork/items2.so")
lib.item_spin.argtypes = [ctypes.c_int64] * 4
lib.item_spin.restype = ctypes.c_int64
launch = ctypes.CDLL("/nvme0n1-disk/code/mojo-jiwer/.parwork/launch.so")
launch.launch_time.argtypes = [ctypes.c_int64] * 2
launch.launch_time.restype = ctypes.c_int64


def py_launch():
    def noop():
        return None

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda _: noop(), range(4)))


py = [() for _ in range(9)]
walls = []
for _ in range(9):
    t = time.perf_counter()
    py_launch()
    walls.append(time.perf_counter() - t)
print("python thread launch, 4 workers: median",
      f"{statistics.median(walls)*1e3:.3f} ms")

mojo = [launch.launch_time(4, 4) / 1e6 for _ in range(15)]
print("max.algorithm parallelize launch, 4 workers: median",
      f"{statistics.median(mojo):.2f} ms")

for spin_us in (100, 500, 1000, 2000, 4000):
    n = 200
    slots = np.zeros(2 * n, dtype=np.int64)
    walls = []
    for _ in range(7):
        t = time.perf_counter()
        lib.item_spin(slots.ctypes.data, n, 4, spin_us * 1000)
        walls.append(time.perf_counter() - t)
    serial_ms = n * spin_us / 1000.0
    wall_ms = statistics.median(walls) * 1e3
    print(f"items {n} x {spin_us}us: serial {serial_ms:.1f} ms, "
          f"wall {wall_ms:.2f} ms, speedup {serial_ms / wall_ms:.2f}x", flush=True)

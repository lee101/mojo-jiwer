import time, statistics
from concurrent.futures import ThreadPoolExecutor
def noop(): return None
def py_launch():
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(lambda _: noop(), range(4)))
walls=[]
for _ in range(15):
    t=time.perf_counter(); py_launch(); walls.append(time.perf_counter()-t)
print(f"python ThreadPoolExecutor launch, 4 workers: median {statistics.median(walls)*1e3:.3f} ms")

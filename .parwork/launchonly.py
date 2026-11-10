import ctypes, statistics
launch = ctypes.CDLL("/nvme0n1-disk/code/mojo-jiwer/.parwork/launch.so")
launch.launch_time.argtypes=[ctypes.c_int64]*2; launch.launch_time.restype=ctypes.c_int64
for w in (2,4):
    vals=[launch.launch_time(4,w)/1e6 for _ in range(15)]
    print(f"max.algorithm parallelize launch, {w} workers: median {statistics.median(vals):.2f} ms, min {min(vals):.2f} ms")

import ctypes, time, statistics
lib=ctypes.CDLL('/nvme0n1-disk/code/mojo-jiwer/.parwork/launch.so')
lib.launch_time.argtypes=[ctypes.c_int64]*2; lib.launch_time.restype=ctypes.c_int64
lib.empty_launch.argtypes=[ctypes.c_int64]*2; lib.empty_launch.restype=ctypes.c_int64
for workers in (1,2,4):
    for n in (4, 400):
        vals=[lib.launch_time(n,workers) for _ in range(9)]
        print("in-mojo launch_time workers",workers,"n",n,"median",statistics.median(vals)/1e6,"ms",flush=True)

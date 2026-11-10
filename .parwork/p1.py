import ctypes, numpy as np, sys, time
lib = ctypes.CDLL(sys.argv[1])
lib.probe_serial.argtypes=[ctypes.c_int64]; lib.probe_serial.restype=ctypes.c_int64
lib.probe_parallel_call.argtypes=[ctypes.c_int64]*3; lib.probe_parallel_call.restype=ctypes.c_int64
buf=np.zeros(1,dtype=np.int64)
print("serial",lib.probe_serial(10),flush=True)
print("par",lib.probe_parallel_call(buf.ctypes.data,1000,2),flush=True)

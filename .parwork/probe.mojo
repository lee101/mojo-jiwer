"""Probe: max.algorithm.parallelize from a shared lib called via ctypes."""

from max.algorithm import parallelize


@export("probe_serial")
def probe_serial(n: Int) abi("C") -> Int:
    var total = 0
    for i in range(n):
        total += i
    return total


@export("probe_parallel_call")
def probe_parallel_call(out_addr: Int, n: Int, workers: Int) abi("C") -> Int:
    var base = out_addr

    def work(chunk_index: Int) {imm}:
        var out = Pointer[Int64, AnyOrigin[mut=True]](unsafe_from_address=base)
        var start = n * chunk_index // workers
        var end = n * (chunk_index + 1) // workers
        for i in range(start, end):
            out.unsafe_store(i, Int64(i * 2))

    parallelize(work, workers, workers)
    return workers

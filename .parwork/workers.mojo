from max.algorithm import parallelize
from std.time import perf_counter_ns

comptime CHUNKS = 24


@export("count_workers")
def count_workers(slots_addr: Int, n: Int, workers: Int, item_ns: Int) abi("C") -> Int:
    var slots = Pointer[Int64, AnyOrigin[mut=True]](unsafe_from_address=slots_addr)

    def work(index: Int) {imm}:
        var t = Int(perf_counter_ns())
        slots.unsafe_store(index, Int64(t))

    parallelize(work, n, workers)
    return n


@export("spin_ns")
def spin_ns(ns: Int) abi("C") -> Int:
    var start = perf_counter_ns()
    while perf_counter_ns() - start < ns:
        pass
    return ns

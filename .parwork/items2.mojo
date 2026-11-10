from max.algorithm import parallelize
from std.time import perf_counter_ns


@export("item_spin")
def item_spin(slots_addr: Int, n: Int, workers: Int, spin: Int) abi("C") -> Int:
    var slots = Pointer[Int64, AnyOrigin[mut=True]](unsafe_from_address=slots_addr)

    def work(index: Int) {imm}:
        var start = perf_counter_ns()
        var t = start
        while perf_counter_ns() - t < spin:
            pass
        slots.unsafe_store(2 * index, Int64(start))
        slots.unsafe_store(2 * index + 1, Int64(perf_counter_ns()))

    parallelize(work, n, workers)
    return n

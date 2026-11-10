from max.algorithm import parallelize
from std.time import perf_counter_ns


@export("empty_launch")
def empty_launch(n: Int, workers: Int) abi("C") -> Int:
    def work(index: Int) {imm}:
        pass

    parallelize(work, n, workers)
    return n


@export("launch_time")
def launch_time(n: Int, workers: Int) abi("C") -> Int:
    var t0 = perf_counter_ns()

    def work(index: Int) {imm}:
        pass

    parallelize(work, n, workers)
    return perf_counter_ns() - t0

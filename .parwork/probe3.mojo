from max.algorithm import parallelize

@export("call_workers")
def call_workers(n: Int, workers: Int) abi("C") -> Int:
    def work(i: Int) {imm}:
        pass
    parallelize(work, n, workers)
    return n

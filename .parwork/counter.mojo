from max.algorithm import parallelize


@export("counter")
def counter(counts_addr: Int, n: Int, workers: Int, chunk: Int) abi("C") -> Int:
    def work(index: Int) {imm}:
        var counts = Pointer[Int64, AnyOrigin[mut=True]](unsafe_from_address=counts_addr)
        var count = counts.unsafe_load(index)
        counts.unsafe_store(index, count + 1)

    parallelize(work, n, workers)
    return n

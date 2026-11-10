from max.algorithm import parallelize


@export("spin_work")
def spin_work(out_addr: Int, n: Int, heavy: Int, workers: Int) abi("C") -> Int:
    def work(i: Int) {imm}:
        var out = Pointer[Int64, AnyOrigin[mut=True]](unsafe_from_address=out_addr)
        var acc = UInt64(0)
        for k in range(heavy):
            acc = acc * 6364136223846793005 + UInt64(k)
        out.unsafe_store(i, Int64(acc))
    parallelize(work, n, workers)
    return n


@export("heavy_call")
def heavy_call(out_addr: Int, n: Int, heavy: Int, workers: Int, chunk: Int) abi("C") -> Int:
    var start = n * chunk // workers
    var end = n * (chunk + 1) // workers

    def work(i: Int) {imm}:
        var out = Pointer[Int64, AnyOrigin[mut=True]](unsafe_from_address=out_addr)
        var acc = UInt64(0)
        for k in range(heavy):
            acc = acc * 6364136223846793005 + UInt64(k)
        out.unsafe_store(i, Int64(acc))
    parallelize(work, end - start, workers)
    return end - start

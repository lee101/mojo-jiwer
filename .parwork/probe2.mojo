from max.algorithm import parallelize

@export("probe_items")
def probe_items(out_addr: Int, n: Int, workers: Int) abi("C") -> Int:
    var base = out_addr

    def work(i: Int) {imm}:
        var out = Pointer[Int64, AnyOrigin[mut=True]](unsafe_from_address=base)
        out.unsafe_store(i, Int64(i * 2))

    parallelize(work, n, workers)
    return n

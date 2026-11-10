from max.algorithm import parallelize

@export("store_items")
def store_items(out_addr: Int, n: Int, workers: Int) abi("C") -> Int:
    var base = out_addr

    def work(i: Int) {imm}:
        var out = Pointer[Int64, AnyOrigin[mut=True]](unsafe_from_address=base)
        out.unsafe_store(i, Int64(i))

    parallelize(work, n, workers)
    return n

@export("store_span")
def store_span(out_addr: Int, n: Int, workers: Int, chunk_index: Int) abi("C") -> Int:
    var base = out_addr
    var start = n * chunk_index // workers
    var end = n * (chunk_index + 1) // workers
    var out = Pointer[Int64, AnyOrigin[mut=True]](unsafe_from_address=base)
    for i in range(start, end):
        out.unsafe_store(i, Int64(i))
    return end - start

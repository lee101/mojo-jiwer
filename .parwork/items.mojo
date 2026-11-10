from max.algorithm import parallelize


@export("item_work")
def item_work(values_addr: Int, n: Int, workers: Int) abi("C") -> Int:
    var values = Pointer[Int64, AnyOrigin[mut=True]](unsafe_from_address=values_addr)
    var chunk = 0

    def work(index: Int) {imm}:
        var slot = index // 64
        if slot >= workers:
            slot = workers - 1
        var target = Pointer[Int64, AnyOrigin[mut=True]](
            unsafe_from_address=values_addr + slot * 1024 * 8
        )
        var acc = UInt64(0)
        for k in range(2000):
            acc = acc * 6364136223846793005 + UInt64(k) + UInt64(index)
        target.unsafe_store(index % 1024, Int64(acc))

    parallelize(work, n, workers)
    return n

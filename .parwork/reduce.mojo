from max.algorithm import parallelize

comptime I64Ptr = Pointer[Int64, AnyOrigin[mut=True]]


def update_one(values: I64Ptr, index: Int):
    var value = values.unsafe_load(index)
    if value % 4 == 0:
        values.unsafe_store(index, -value)


@export("par_update")
def par_update(values_addr: Int, n: Int, workers: Int) abi("C") -> Int:
    var values = I64Ptr(unsafe_from_address=values_addr)

    def work(index: Int) {imm}:
        update_one(values, index)

    parallelize(work, n, workers)
    var total = 0
    for i in range(n):
        total += Int(values.unsafe_load(i))
    return total

@export("echo_args")
def echo_args(
    result_addr: Int,
    a: Int,
    b: Int,
    c: Int,
    d: Int,
    e: Int,
    f: Int,
    g: Int,
    h: Int,
    i: Int,
) abi("C") -> Int:
    var result = Pointer[Int64, AnyOrigin[mut=True]](unsafe_from_address=result_addr)
    result.unsafe_store(0, Int64(a))
    result.unsafe_store(1, Int64(b))
    result.unsafe_store(2, Int64(c))
    result.unsafe_store(3, Int64(d))
    result.unsafe_store(4, Int64(e))
    result.unsafe_store(5, Int64(f))
    result.unsafe_store(6, Int64(g))
    result.unsafe_store(7, Int64(h))
    result.unsafe_store(8, Int64(i))
    return 9

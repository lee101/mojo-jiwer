from max.algorithm import parallelize

@export("bad1")
def bad1(n: Int) abi("C") -> Int:
    def work(i: Int) {imm}:
        pass
    parallelize(work, "hello")
    return n

@export("bad2")
def bad2(n: Int) abi("C") -> Int:
    def work(i: Int) {imm}:
        pass
    parallelize(work, n, 4, "x")
    return n

"""Probe: real Myers work, serial vs max.algorithm.parallelize, same helper."""

from max.algorithm import parallelize


def matching_prefix(left: Pointer[Int64, AnyOrigin[mut=True]], right: Pointer[Int64, AnyOrigin[mut=True]], limit: Int) -> Int:
    var index = 0
    while index < limit and left.unsafe_load(index) == right.unsafe_load(index):
        index += 1
    return index


def myers_64(
    pattern: Pointer[Int64, AnyOrigin[mut=True]],
    text: Pointer[Int64, AnyOrigin[mut=True]],
    pattern_len: Int,
    text_len: Int,
    masks: Pointer[UInt64, AnyOrigin[mut=True]],
    stride: Int,
) -> Int:
    for pi in range(pattern_len):
        masks.unsafe_store(Int(pattern.unsafe_load(pi)) * stride, UInt64(0))
    for pi in range(pattern_len):
        var offset = Int(pattern.unsafe_load(pi)) * stride
        masks.unsafe_store(offset, masks.unsafe_load(offset) | (UInt64(1) << UInt64(pi)))
    var positive = ~UInt64(0)
    var negative = UInt64(0)
    var score = pattern_len
    var last = UInt64(1) << UInt64(pattern_len - 1)
    for ti in range(text_len):
        var equal = masks.unsafe_load(Int(text.unsafe_load(ti)) * stride)
        var xv = equal | negative
        var xh = (((equal & positive) + positive) ^ positive) | equal
        var ph = negative | ~(xh | positive)
        var mh = positive & xh
        if ph & last:
            score += 1
        elif mh & last:
            score -= 1
        ph = (ph << 1) | UInt64(1)
        mh <<= 1
        positive = mh | ~(xv | ph)
        negative = ph & xv
    for pi in range(pattern_len):
        masks.unsafe_store(Int(pattern.unsafe_load(pi)) * stride, UInt64(0))
    return score


def one(
    refs_addr: Int,
    hyps_addr: Int,
    ro_addr: Int,
    ho_addr: Int,
    result_addr: Int,
    masks_addr: Int,
    stride: Int,
    index: Int,
):
    var refs = Pointer[Int64, AnyOrigin[mut=True]](unsafe_from_address=refs_addr)
    var hyps = Pointer[Int64, AnyOrigin[mut=True]](unsafe_from_address=hyps_addr)
    var ro = Pointer[Int64, AnyOrigin[mut=True]](unsafe_from_address=ro_addr)
    var ho = Pointer[Int64, AnyOrigin[mut=True]](unsafe_from_address=ho_addr)
    var result = Pointer[Int64, AnyOrigin[mut=True]](unsafe_from_address=result_addr)
    var masks = Pointer[UInt64, AnyOrigin[mut=True]](unsafe_from_address=masks_addr)
    var rs = Int(ro.unsafe_load(index))
    var hs = Int(ho.unsafe_load(index))
    var rl = Int(ro.unsafe_load(index + 1)) - rs
    var hl = Int(ho.unsafe_load(index + 1)) - hs
    var a = refs.unsafe_offset(rs)
    var b = hyps.unsafe_offset(hs)
    var best = 0
    if rl >= hl:
        var p = matching_prefix(a, b, hl)
        best = myers_64(b.unsafe_offset(p), a.unsafe_offset(p), hl - p, rl - p, masks, stride)
    else:
        var p = matching_prefix(b, a, rl)
        best = myers_64(a.unsafe_offset(p), b.unsafe_offset(p), rl - p, hl - p, masks, stride)
    result.unsafe_store(index, Int64(best))


@export("k_serial")
def k_serial(
    refs_addr: Int,
    hyps_addr: Int,
    ro_addr: Int,
    ho_addr: Int,
    result_addr: Int,
    masks_addr: Int,
    stride: Int,
    count: Int,
    chunk: Int,
    workers: Int,
) abi("C") -> Int:
    var masks = Pointer[UInt64, AnyOrigin[mut=True]](unsafe_from_address=masks_addr)
    for index in range(count):
        one(refs_addr, hyps_addr, ro_addr, ho_addr, result_addr, masks_addr, stride, index)
    return count


@export("k_parallel")
def k_parallel(
    refs_addr: Int,
    hyps_addr: Int,
    ro_addr: Int,
    ho_addr: Int,
    result_addr: Int,
    masks_addr: Int,
    stride: Int,
    count: Int,
    chunk: Int,
    workers: Int,
) abi("C") -> Int:
    def work(index: Int) {imm}:
        one(refs_addr, hyps_addr, ro_addr, ho_addr, result_addr, masks_addr, stride, index)

    parallelize(work, count, workers)
    return count

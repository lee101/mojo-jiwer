"""Levenshtein kernels exposed through a stable C ABI."""

from std.algorithm import parallelize
from std.math import iota
from std.sys.info import simd_width_of

comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime I32Ptr = UnsafePointer[Int32, AnyOrigin[mut=True]]
comptime U64Ptr = UnsafePointer[UInt64, AnyOrigin[mut=True]]
comptime U8Ptr = UnsafePointer[UInt8, AnyOrigin[mut=True]]


def i64_ptr(addr: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=addr)


def i32_ptr(addr: Int) -> I32Ptr:
    return I32Ptr(unsafe_from_address=addr)


def u64_ptr(addr: Int) -> U64Ptr:
    return U64Ptr(unsafe_from_address=addr)


def u8_ptr(addr: Int) -> U8Ptr:
    return U8Ptr(unsafe_from_address=addr)


def matching_prefix(left: I64Ptr, right: I64Ptr, limit: Int) -> Int:
    comptime W = simd_width_of[DType.float64]()
    var index = 0
    while index + W <= limit:
        var matches = (
            left.load[width=W](index).eq(right.load[width=W](index))
        ).select(
            SIMD[DType.int64, W](1),
            SIMD[DType.int64, W](0),
        )
        if Int(matches.reduce_add()) != W:
            break
        index += W
    while index < limit and left.load(index) == right.load(index):
        index += 1
    return index


def matching_suffix(
    left: I64Ptr,
    right: I64Ptr,
    left_len: Int,
    right_len: Int,
    limit: Int,
) -> Int:
    comptime W = simd_width_of[DType.float64]()
    var count = 0
    while count + W <= limit:
        var left_start = left_len - count - W
        var right_start = right_len - count - W
        var matches = (
            left.load[width=W](left_start).eq(
                right.load[width=W](right_start)
            )
        )
        if not Bool(matches.reduce_and()):
            break
        count += W
    while (
        count < limit
        and left.load(left_len - count - 1)
        == right.load(right_len - count - 1)
    ):
        count += 1
    return count


def myers_64(
    pattern: I64Ptr,
    text: I64Ptr,
    pattern_start: Int,
    text_start: Int,
    pattern_len: Int,
    text_len: Int,
    masks: U64Ptr,
) -> Int:
    for pi in range(pattern_len):
        masks.store(Int(pattern.load(pattern_start + pi)), UInt64(0))
    for pi in range(pattern_len):
        var token = Int(pattern.load(pattern_start + pi))
        masks.store(token, masks.load(token) | (UInt64(1) << UInt64(pi)))

    var positive = ~UInt64(0)
    var negative = UInt64(0)
    var score = pattern_len
    var last = UInt64(1) << UInt64(pattern_len - 1)

    for ti in range(text_len):
        var equal = masks.load(Int(text.load(text_start + ti)))
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
        masks.store(Int(pattern.load(pattern_start + pi)), UInt64(0))
    return score


def distance_impl(
    reference: I64Ptr,
    hypothesis: I64Ptr,
    reference_len: Int,
    hypothesis_len: Int,
    rows: I32Ptr,
    masks: U64Ptr,
) -> Int:
    var shared_prefix = matching_prefix(
        reference, hypothesis, min(reference_len, hypothesis_len)
    )
    var ref_start = shared_prefix
    var hyp_start = shared_prefix
    var n = reference_len
    var m = hypothesis_len

    n -= shared_prefix
    m -= shared_prefix
    var shared_suffix = matching_suffix(
        reference + ref_start,
        hypothesis + hyp_start,
        n,
        m,
        min(n, m),
    )
    n -= shared_suffix
    m -= shared_suffix

    if n == 0:
        return m
    if m == 0:
        return n

    if m <= 63:
        return myers_64(hypothesis, reference, hyp_start, ref_start, m, n, masks)
    if n <= 63:
        return myers_64(reference, hypothesis, ref_start, hyp_start, n, m, masks)

    var width = m + 1
    comptime W = simd_width_of[DType.float64]()
    var j = 0
    while j + W <= width:
        rows.store(j, iota[DType.int32, W](Int32(j)))
        j += W
    while j < width:
        rows.store(j, Int32(j))
        j += 1

    for i in range(1, n + 1):
        var previous_offset = ((i - 1) & 1) * width
        var current_offset = (i & 1) * width
        rows.store(current_offset, Int32(i))
        var ref_token = reference.load(ref_start + i - 1)
        for j in range(1, m + 1):
            var diagonal = rows.load(previous_offset + j - 1)
            var deletion = rows.load(previous_offset + j) + 1
            var insertion = rows.load(current_offset + j - 1) + 1
            var substitution = diagonal
            if ref_token != hypothesis.load(hyp_start + j - 1):
                substitution += 1
            rows.store(current_offset + j, min(substitution, min(deletion, insertion)))

    return Int(rows.load((n & 1) * width + m))


@export("mji_distance")
def mji_distance(
    reference_addr: Int,
    hypothesis_addr: Int,
    reference_len: Int,
    hypothesis_len: Int,
    rows_addr: Int,
    masks_addr: Int,
) abi("C") -> Int:
    return distance_impl(
        i64_ptr(reference_addr),
        i64_ptr(hypothesis_addr),
        reference_len,
        hypothesis_len,
        i32_ptr(rows_addr),
        u64_ptr(masks_addr),
    )


@export("mji_distances")
def mji_distances(
    references_addr: Int,
    hypotheses_addr: Int,
    reference_offsets_addr: Int,
    hypothesis_offsets_addr: Int,
    count: Int,
    worker_count: Int,
    rows_addr: Int,
    row_stride: Int,
    masks_addr: Int,
    mask_stride: Int,
    distances_addr: Int,
) abi("C"):
    var references = i64_ptr(references_addr)
    var hypotheses = i64_ptr(hypotheses_addr)
    var reference_offsets = i64_ptr(reference_offsets_addr)
    var hypothesis_offsets = i64_ptr(hypothesis_offsets_addr)
    var rows = i32_ptr(rows_addr)
    var masks = u64_ptr(masks_addr)
    var distances = i64_ptr(distances_addr)
    @parameter
    def compute(index: Int, scratch_index: Int):
        var ref_start = Int(reference_offsets.load(index))
        var hyp_start = Int(hypothesis_offsets.load(index))
        var ref_len = Int(reference_offsets.load(index + 1)) - ref_start
        var hyp_len = Int(hypothesis_offsets.load(index + 1)) - hyp_start
        distances.store(
            index,
            Int64(
                distance_impl(
                    references + ref_start,
                    hypotheses + hyp_start,
                    ref_len,
                    hyp_len,
                    rows + scratch_index * row_stride,
                    masks + scratch_index * mask_stride,
                )
            ),
        )

    @parameter
    def compute_chunk(chunk: Int):
        var start = count * chunk // worker_count
        var end = count * (chunk + 1) // worker_count
        for index in range(start, end):
            compute(index, chunk)

    if worker_count > 1:
        parallelize[compute_chunk](worker_count, worker_count)
    else:
        for index in range(count):
            compute(index, 0)


def trace_impl(
    reference: I64Ptr,
    hypothesis: I64Ptr,
    reference_len: Int,
    hypothesis_len: Int,
    matrix: I32Ptr,
    operations: U8Ptr,
) -> Int:
    var prefix = matching_prefix(
        reference, hypothesis, min(reference_len, hypothesis_len)
    )
    var suffix = matching_suffix(
        reference,
        hypothesis,
        reference_len,
        hypothesis_len,
        min(reference_len - prefix, hypothesis_len - prefix),
    )

    var n = reference_len - prefix - suffix
    var m = hypothesis_len - prefix - suffix
    var width = m + 1

    for i in range(n + 1):
        matrix.store(i * width, Int32(i))
    comptime W = simd_width_of[DType.float64]()
    var boundary_j = 1
    while boundary_j + W <= m + 1:
        matrix.store(
            boundary_j,
            iota[DType.int32, W](Int32(boundary_j)),
        )
        boundary_j += W
    while boundary_j < m + 1:
        matrix.store(boundary_j, Int32(boundary_j))
        boundary_j += 1

    for i in range(1, n + 1):
        var ref_token = reference.load(prefix + i - 1)
        for j in range(1, m + 1):
            var diagonal = matrix.load((i - 1) * width + j - 1)
            var deletion = matrix.load((i - 1) * width + j) + 1
            var insertion = matrix.load(i * width + j - 1) + 1
            var substitution = diagonal
            if ref_token != hypothesis.load(prefix + j - 1):
                substitution += 1
            matrix.store(i * width + j, min(substitution, min(deletion, insertion)))

    var i = n
    var j = m
    var count = suffix
    for k in range(suffix):
        operations.store(k, UInt8(0))

    while i > 0 and j > 0:
        var current = matrix.load(i * width + j)
        if current == matrix.load((i - 1) * width + j) + 1:
            operations.store(count, UInt8(2))
            i -= 1
        else:
            j -= 1
            if (
                j > 0
                and matrix.load(i * width + j)
                == matrix.load((i - 1) * width + j) - 1
            ):
                operations.store(count, UInt8(3))
            else:
                i -= 1
                if (
                    reference.load(prefix + i)
                    == hypothesis.load(prefix + j)
                ):
                    operations.store(count, UInt8(0))
                else:
                    operations.store(count, UInt8(1))
        count += 1

    while i > 0:
        operations.store(count, UInt8(2))
        count += 1
        i -= 1
    while j > 0:
        operations.store(count, UInt8(3))
        count += 1
        j -= 1

    for k in range(prefix):
        operations.store(count + k, UInt8(0))
    count += prefix

    return count


@export("mji_trace")
def mji_trace(
    reference_addr: Int,
    hypothesis_addr: Int,
    reference_len: Int,
    hypothesis_len: Int,
    matrix_addr: Int,
    operations_addr: Int,
) abi("C") -> Int:
    return trace_impl(
        i64_ptr(reference_addr),
        i64_ptr(hypothesis_addr),
        reference_len,
        hypothesis_len,
        i32_ptr(matrix_addr),
        u8_ptr(operations_addr),
    )

"""Levenshtein kernels exposed through a stable C ABI."""

from std.math import iota
from std.runtime import initialize_runtime
from std.runtime.asyncrt import TaskGroup
from std.sys.info import simd_width_of

comptime I64Ptr = Pointer[Int64, AnyOrigin[mut=True]]
comptime I32Ptr = Pointer[Int32, AnyOrigin[mut=True]]
comptime U64Ptr = Pointer[UInt64, AnyOrigin[mut=True]]
comptime U8Ptr = Pointer[UInt8, AnyOrigin[mut=True]]


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
            left.unsafe_load[width=W](index).eq(right.unsafe_load[width=W](index))
        ).select(
            SIMD[DType.int64, W](1),
            SIMD[DType.int64, W](0),
        )
        if Int(matches.reduce_add()) != W:
            break
        index += W
    while index < limit and left.unsafe_load(index) == right.unsafe_load(index):
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
            left.unsafe_load[width=W](left_start).eq(
                right.unsafe_load[width=W](right_start)
            )
        )
        if not Bool(matches.reduce_and()):
            break
        count += W
    while (
        count < limit
        and left.unsafe_load(left_len - count - 1)
        == right.unsafe_load(right_len - count - 1)
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
        masks.unsafe_store(Int(pattern.unsafe_load(pattern_start + pi)), UInt64(0))
    for pi in range(pattern_len):
        var token = Int(pattern.unsafe_load(pattern_start + pi))
        masks.unsafe_store(token, masks.unsafe_load(token) | (UInt64(1) << UInt64(pi)))

    var positive = ~UInt64(0)
    var negative = UInt64(0)
    var score = pattern_len
    var last = UInt64(1) << UInt64(pattern_len - 1)

    for ti in range(text_len):
        var equal = masks.unsafe_load(Int(text.unsafe_load(text_start + ti)))
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
        masks.unsafe_store(Int(pattern.unsafe_load(pattern_start + pi)), UInt64(0))
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
        reference.unsafe_offset(ref_start),
        hypothesis.unsafe_offset(hyp_start),
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
        rows.unsafe_store(j, iota[DType.int32, W](Int32(j)))
        j += W
    while j < width:
        rows.unsafe_store(j, Int32(j))
        j += 1

    for i in range(1, n + 1):
        var previous_offset = ((i - 1) & 1) * width
        var current_offset = (i & 1) * width
        rows.unsafe_store(current_offset, Int32(i))
        var ref_token = reference.unsafe_load(ref_start + i - 1)
        for j in range(1, m + 1):
            var diagonal = rows.unsafe_load(previous_offset + j - 1)
            var deletion = rows.unsafe_load(previous_offset + j) + 1
            var insertion = rows.unsafe_load(current_offset + j - 1) + 1
            var substitution = diagonal
            if ref_token != hypothesis.unsafe_load(hyp_start + j - 1):
                substitution += 1
            rows.unsafe_store(current_offset + j, min(substitution, min(deletion, insertion)))

    return Int(rows.unsafe_load((n & 1) * width + m))


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


def compute_distance_at(
    references: I64Ptr,
    hypotheses: I64Ptr,
    reference_offsets: I64Ptr,
    hypothesis_offsets: I64Ptr,
    rows: I32Ptr,
    row_stride: Int,
    masks: U64Ptr,
    mask_stride: Int,
    distances: I64Ptr,
    index: Int,
    scratch_index: Int,
):
    var ref_start = Int(reference_offsets.unsafe_load(index))
    var hyp_start = Int(hypothesis_offsets.unsafe_load(index))
    var ref_len = Int(reference_offsets.unsafe_load(index + 1)) - ref_start
    var hyp_len = Int(hypothesis_offsets.unsafe_load(index + 1)) - hyp_start
    distances.unsafe_store(
        index,
        Int64(
            distance_impl(
                references.unsafe_offset(ref_start),
                hypotheses.unsafe_offset(hyp_start),
                ref_len,
                hyp_len,
                rows.unsafe_offset(scratch_index * row_stride),
                masks.unsafe_offset(scratch_index * mask_stride),
            )
        ),
    )


async def compute_distance_chunk(
    references: I64Ptr,
    hypotheses: I64Ptr,
    reference_offsets: I64Ptr,
    hypothesis_offsets: I64Ptr,
    count: Int,
    worker_count: Int,
    rows: I32Ptr,
    row_stride: Int,
    masks: U64Ptr,
    mask_stride: Int,
    distances: I64Ptr,
    chunk: Int,
):
    var start = count * chunk // worker_count
    var end = count * (chunk + 1) // worker_count
    for index in range(start, end):
        compute_distance_at(
            references,
            hypotheses,
            reference_offsets,
            hypothesis_offsets,
            rows,
            row_stride,
            masks,
            mask_stride,
            distances,
            index,
            chunk,
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

    if worker_count > 1:
        initialize_runtime()
        var tasks = TaskGroup()
        for chunk in range(worker_count):
            tasks.create_task(
                compute_distance_chunk(
                    references,
                    hypotheses,
                    reference_offsets,
                    hypothesis_offsets,
                    count,
                    worker_count,
                    rows,
                    row_stride,
                    masks,
                    mask_stride,
                    distances,
                    chunk,
                )
            )
        tasks.wait()
    else:
        for index in range(count):
            compute_distance_at(
                references,
                hypotheses,
                reference_offsets,
                hypothesis_offsets,
                rows,
                row_stride,
                masks,
                mask_stride,
                distances,
                index,
                0,
            )


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
        matrix.unsafe_store(i * width, Int32(i))
    comptime W = simd_width_of[DType.float64]()
    var boundary_j = 1
    while boundary_j + W <= m + 1:
        matrix.unsafe_store(
            boundary_j,
            iota[DType.int32, W](Int32(boundary_j)),
        )
        boundary_j += W
    while boundary_j < m + 1:
        matrix.unsafe_store(boundary_j, Int32(boundary_j))
        boundary_j += 1

    for i in range(1, n + 1):
        var ref_token = reference.unsafe_load(prefix + i - 1)
        for j in range(1, m + 1):
            var diagonal = matrix.unsafe_load((i - 1) * width + j - 1)
            var deletion = matrix.unsafe_load((i - 1) * width + j) + 1
            var insertion = matrix.unsafe_load(i * width + j - 1) + 1
            var substitution = diagonal
            if ref_token != hypothesis.unsafe_load(prefix + j - 1):
                substitution += 1
            matrix.unsafe_store(i * width + j, min(substitution, min(deletion, insertion)))

    var i = n
    var j = m
    var count = suffix
    for k in range(suffix):
        operations.unsafe_store(k, UInt8(0))

    while i > 0 and j > 0:
        var current = matrix.unsafe_load(i * width + j)
        if current == matrix.unsafe_load((i - 1) * width + j) + 1:
            operations.unsafe_store(count, UInt8(2))
            i -= 1
        else:
            j -= 1
            if (
                j > 0
                and matrix.unsafe_load(i * width + j)
                == matrix.unsafe_load((i - 1) * width + j) - 1
            ):
                operations.unsafe_store(count, UInt8(3))
            else:
                i -= 1
                if (
                    reference.unsafe_load(prefix + i)
                    == hypothesis.unsafe_load(prefix + j)
                ):
                    operations.unsafe_store(count, UInt8(0))
                else:
                    operations.unsafe_store(count, UInt8(1))
        count += 1

    while i > 0:
        operations.unsafe_store(count, UInt8(2))
        count += 1
        i -= 1
    while j > 0:
        operations.unsafe_store(count, UInt8(3))
        count += 1
        j -= 1

    for k in range(prefix):
        operations.unsafe_store(count + k, UInt8(0))
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

from max.algorithm import parallelize
from std.sys.info import simd_width_of

comptime I64Ptr = Pointer[Int64, AnyOrigin[mut=True]]
comptime U64Ptr = Pointer[UInt64, AnyOrigin[mut=True]]


def matching_prefix(left: I64Ptr, right: I64Ptr, limit: Int) -> Int:
    var index = 0
    while index < limit and left.unsafe_load(index) == right.unsafe_load(index):
        index += 1
    return index


def matching_suffix(left: I64Ptr, right: I64Ptr, left_len: Int, right_len: Int, limit: Int) -> Int:
    var count = 0
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
    mask_word_stride: Int,
) -> Int:
    for pi in range(pattern_len):
        var token = Int(pattern.unsafe_load(pattern_start + pi))
        masks.unsafe_store(token * mask_word_stride, UInt64(0))
    for pi in range(pattern_len):
        var token = Int(pattern.unsafe_load(pattern_start + pi))
        var offset = token * mask_word_stride
        masks.unsafe_store(
            offset,
            masks.unsafe_load(offset) | (UInt64(1) << UInt64(pi)),
        )

    var positive = ~UInt64(0)
    var negative = UInt64(0)
    var score = pattern_len
    var last = UInt64(1) << UInt64(pattern_len - 1)

    for ti in range(text_len):
        var equal = masks.unsafe_load(
            Int(text.unsafe_load(text_start + ti)) * mask_word_stride
        )
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
        masks.unsafe_store(
            Int(pattern.unsafe_load(pattern_start + pi)) * mask_word_stride,
            UInt64(0),
        )
    return score




def fill_words(pointer: U64Ptr, count: Int, value: UInt64):
    comptime W = simd_width_of[DType.float64]()
    var values = SIMD[DType.uint64, W](value)
    var i = 0
    while i + W <= count:
        pointer.unsafe_store(i, values)
        i += W
    while i < count:
        pointer.unsafe_store(i, value)
        i += 1




def myers_multiword(
    pattern: I64Ptr,
    text: I64Ptr,
    pattern_start: Int,
    text_start: Int,
    pattern_len: Int,
    text_len: Int,
    masks: U64Ptr,
    mask_word_stride: Int,
    mask_state_offset: Int,
) -> Int:
    var words = (pattern_len + 63) // 64
    for pi in range(pattern_len):
        var token = Int(pattern.unsafe_load(pattern_start + pi))
        masks.unsafe_store(
            token * mask_word_stride + pi // 64,
            UInt64(0),
        )
    for pi in range(pattern_len):
        var token = Int(pattern.unsafe_load(pattern_start + pi))
        var offset = token * mask_word_stride + pi // 64
        masks.unsafe_store(
            offset,
            masks.unsafe_load(offset) | (UInt64(1) << UInt64(pi % 64)),
        )

    var positive = masks.unsafe_offset(mask_state_offset)
    var negative = positive.unsafe_offset(words)
    fill_words(positive, words, ~UInt64(0))
    fill_words(negative, words, UInt64(0))

    var score = pattern_len
    var final_word = words - 1
    var high_bit = UInt64(1) << UInt64((pattern_len - 1) % 64)
    for ti in range(text_len):
        var add_carry = UInt64(0)
        var positive_carry = UInt64(1)
        var negative_carry = UInt64(0)
        var token_offset = (
            Int(text.unsafe_load(text_start + ti)) * mask_word_stride
        )
        for word in range(words):
            var pv = positive.unsafe_load(word)
            var nv = negative.unsafe_load(word)
            var equal = masks.unsafe_load(token_offset + word)
            var xv = equal | nv
            var left = xv & pv
            var first_sum = left + pv
            var first_carry = first_sum < left
            var full_sum = first_sum + add_carry
            var second_carry = full_sum < first_sum
            add_carry = UInt64(first_carry or second_carry)
            var xh = (full_sum ^ pv) | xv
            var ph = nv | ~(xh | pv)
            var mh = xh & pv
            if word == final_word:
                if ph & high_bit:
                    score += 1
                elif mh & high_bit:
                    score -= 1
            var next_positive_carry = ph >> 63
            var next_negative_carry = mh >> 63
            ph = (ph << 1) | positive_carry
            mh = (mh << 1) | negative_carry
            positive_carry = next_positive_carry
            negative_carry = next_negative_carry
            positive.unsafe_store(word, mh | ~(xh | ph))
            negative.unsafe_store(word, ph & xh)

    for pi in range(pattern_len):
        masks.unsafe_store(
            Int(pattern.unsafe_load(pattern_start + pi)) * mask_word_stride
            + pi // 64,
            UInt64(0),
        )
    return score



def distance_impl(
    reference: I64Ptr,
    hypothesis: I64Ptr,
    reference_len: Int,
    hypothesis_len: Int,
    masks: U64Ptr,
    mask_word_stride: Int,
    mask_state_offset: Int,
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
        return myers_64(
            hypothesis,
            reference,
            hyp_start,
            ref_start,
            m,
            n,
            masks,
            mask_word_stride,
        )
    if n <= 63:
        return myers_64(
            reference,
            hypothesis,
            ref_start,
            hyp_start,
            n,
            m,
            masks,
            mask_word_stride,
        )

    return myers_multiword(
        hypothesis,
        reference,
        hyp_start,
        ref_start,
        m,
        n,
        masks,
        mask_word_stride,
        mask_state_offset,
    )



def one(
    refs: I64Ptr,
    hyps: I64Ptr,
    ro: I64Ptr,
    ho: I64Ptr,
    result: I64Ptr,
    masks: U64Ptr,
    stride: Int,
    state: Int,
    index: Int,
):
    var rs = Int(ro.unsafe_load(index))
    var hs = Int(ho.unsafe_load(index))
    var rl = Int(ro.unsafe_load(index + 1)) - rs
    var hl = Int(ho.unsafe_load(index + 1)) - hs
    var value = distance_impl(
        refs.unsafe_offset(rs),
        hyps.unsafe_offset(hs),
        rl,
        hl,
        masks,
        stride,
        state,
    )
    result.unsafe_store(index, Int64(value))


@export("k_serial")
def k_serial(
    refs_addr: Int,
    hyps_addr: Int,
    ro_addr: Int,
    ho_addr: Int,
    result_addr: Int,
    masks_addr: Int,
    stride: Int,
    state: Int,
    count: Int,
    workers: Int,
) abi("C") -> Int:
    var refs = I64Ptr(unsafe_from_address=refs_addr)
    var hyps = I64Ptr(unsafe_from_address=hyps_addr)
    var ro = I64Ptr(unsafe_from_address=ro_addr)
    var ho = I64Ptr(unsafe_from_address=ho_addr)
    var result = I64Ptr(unsafe_from_address=result_addr)
    var masks = U64Ptr(unsafe_from_address=masks_addr)
    for index in range(count):
        one(refs, hyps, ro, ho, result, masks, stride, state, index)
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
    state: Int,
    count: Int,
    workers: Int,
) abi("C") -> Int:
    var refs = I64Ptr(unsafe_from_address=refs_addr)
    var hyps = I64Ptr(unsafe_from_address=hyps_addr)
    var ro = I64Ptr(unsafe_from_address=ro_addr)
    var ho = I64Ptr(unsafe_from_address=ho_addr)
    var result = I64Ptr(unsafe_from_address=result_addr)
    var masks = U64Ptr(unsafe_from_address=masks_addr)

    def work(index: Int) {imm}:
        one(refs, hyps, ro, ho, result, masks, stride, state, index)

    parallelize(work, count, workers)
    return count

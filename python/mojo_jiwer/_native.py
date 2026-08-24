"""ctypes bridge to the Mojo edit-distance kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess
from itertools import chain

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIBRARY = os.environ.get("MOJO_JIWER_LIB") or os.path.join(
    ROOT, "dist", "libmojo-jiwer.so"
)

I64 = ctypes.c_int64

_SIGNATURES = {
    "mji_distance": ([I64] * 7, I64),
    "mji_distances": ([I64] * 11, None),
    "mji_trace": ([I64, I64, I64, I64, I64, I64], I64),
    "mji_traces": ([I64] * 11, None),
}

_library: ctypes.CDLL | None = None


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "capi.mojo")
    if not force and os.path.exists(LIBRARY):
        if not os.path.exists(source) or os.path.getmtime(LIBRARY) >= os.path.getmtime(source):
            return LIBRARY
    if os.environ.get("MOJO_JIWER_LIB"):
        raise BuildError(f"MOJO_JIWER_LIB does not exist: {LIBRARY}")
    script = os.path.join(ROOT, "build", "build.sh")
    proc = subprocess.run(
        ["bash", script], cwd=ROOT, capture_output=True, text=True, timeout=1800
    )
    if proc.returncode or not os.path.exists(LIBRARY):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIBRARY


def library() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def _address(array: np.ndarray) -> int:
    address = int(array.ctypes.data)
    if not address:
        raise ValueError("cannot pass a null NumPy buffer to Mojo")
    return address


def _require_i64_vector(name: str, value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != np.dtype(np.int64):
        raise TypeError(f"{name} must have dtype int64, got {value.dtype}")
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def _validate_tokens(name: str, value: np.ndarray) -> None:
    if value.size and int(value.min()) < 0:
        raise ValueError(f"{name} token IDs must be non-negative")


def token_array(values: list[int]) -> np.ndarray:
    if values:
        if any(not isinstance(value, (int, np.integer)) for value in values):
            raise TypeError("token IDs must be integers")
        result = np.asarray(values, dtype=np.int64)
        _validate_tokens("tokens", result)
        return result
    return np.zeros(1, dtype=np.int64)


def distance(reference: list[int], hypothesis: list[int]) -> int:
    ref = token_array(reference)
    hyp = token_array(hypothesis)
    if len(hypothesis) > len(reference):
        ref, hyp = hyp, ref
        reference, hypothesis = hypothesis, reference
    mask_word_stride = max(1, (len(hypothesis) + 63) // 64)
    mask_state_offset = (
        max(int(ref.max()), int(hyp.max())) + 1
    ) * mask_word_stride
    masks = np.zeros(mask_state_offset + 2 * mask_word_stride, dtype=np.uint64)
    return int(
        library().mji_distance(
            _address(ref),
            _address(hyp),
            len(reference),
            len(hypothesis),
            _address(masks),
            mask_word_stride,
            mask_state_offset,
        )
    )


def _flatten_sequences(
    references: list[list[int]], hypotheses: list[list[int]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(references)
    ref_offsets = np.empty(count + 1, dtype=np.int64)
    hyp_offsets = np.empty(count + 1, dtype=np.int64)
    ref_offsets[0] = hyp_offsets[0] = 0
    np.cumsum(
        np.fromiter(map(len, references), dtype=np.int64, count=count),
        out=ref_offsets[1:],
    )
    np.cumsum(
        np.fromiter(map(len, hypotheses), dtype=np.int64, count=count),
        out=hyp_offsets[1:],
    )
    ref_count = int(ref_offsets[-1])
    hyp_count = int(hyp_offsets[-1])
    ref = np.fromiter(
        chain.from_iterable(references), dtype=np.int64, count=ref_count
    )
    hyp = np.fromiter(
        chain.from_iterable(hypotheses), dtype=np.int64, count=hyp_count
    )
    if not ref_count:
        ref = np.zeros(1, dtype=np.int64)
    if not hyp_count:
        hyp = np.zeros(1, dtype=np.int64)
    return ref, hyp, ref_offsets, hyp_offsets


def distances(
    references: list[list[int]], hypotheses: list[list[int]]
) -> list[int]:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must contain the same number of items")
    if not references:
        return []
    pairs = [
        (reference, hypothesis)
        if len(hypothesis) <= len(reference)
        else (hypothesis, reference)
        for reference, hypothesis in zip(references, hypotheses)
    ]
    ref, hyp, ref_offsets, hyp_offsets = _flatten_sequences(
        [pair[0] for pair in pairs], [pair[1] for pair in pairs]
    )
    return distances_flat(ref, hyp, ref_offsets, hyp_offsets)


def distances_flat(
    ref: np.ndarray,
    hyp: np.ndarray,
    ref_offsets: np.ndarray,
    hyp_offsets: np.ndarray,
) -> list[int]:
    ref = _require_i64_vector("ref", ref)
    hyp = _require_i64_vector("hyp", hyp)
    ref_offsets = _require_i64_vector("ref_offsets", ref_offsets)
    hyp_offsets = _require_i64_vector("hyp_offsets", hyp_offsets)
    _validate_tokens("ref", ref)
    _validate_tokens("hyp", hyp)
    if len(ref_offsets) != len(hyp_offsets):
        raise ValueError("offset arrays must have the same length")
    if not len(ref_offsets):
        raise ValueError("offset arrays must contain at least the initial zero")
    for name, offsets, size in (
        ("ref_offsets", ref_offsets, len(ref)),
        ("hyp_offsets", hyp_offsets, len(hyp)),
    ):
        if int(offsets[0]) != 0:
            raise ValueError(f"{name} must start at zero")
        if np.any(offsets[1:] < offsets[:-1]):
            raise ValueError(f"{name} must be monotonically non-decreasing")
        logical_size = int(offsets[-1])
        if logical_size != size and not (logical_size == 0 and size == 1):
            raise ValueError(f"{name} must end at the corresponding buffer length")
    count = len(ref_offsets) - 1
    if not count:
        return []
    if not len(ref):
        ref = np.zeros(1, dtype=np.int64)
    if not len(hyp):
        hyp = np.zeros(1, dtype=np.int64)
    ref_lengths = np.diff(ref_offsets)
    hyp_lengths = np.diff(hyp_offsets)
    mask_word_stride = max(1, (int(hyp_lengths.max(initial=0)) + 63) // 64)
    mask_state_offset = (
        max(int(ref.max()), int(hyp.max())) + 1
    ) * mask_word_stride
    mask_stride = mask_state_offset + 2 * mask_word_stride
    token_work = int(np.maximum(ref_lengths, hyp_lengths).sum())
    worker_count = min(16, count) if count >= 256 and token_work >= 200_000 else 1
    scratch_count = worker_count
    masks = np.zeros(mask_stride * scratch_count, dtype=np.uint64)
    result = np.empty(count, dtype=np.int64)
    library().mji_distances(
        _address(ref),
        _address(hyp),
        _address(ref_offsets),
        _address(hyp_offsets),
        count,
        worker_count,
        _address(masks),
        mask_stride if scratch_count > 1 else 0,
        mask_word_stride,
        mask_state_offset,
        _address(result),
    )
    return result.tolist()


def trace(reference: list[int], hypothesis: list[int]) -> list[int]:
    ref = token_array(reference)
    hyp = token_array(hypothesis)
    matrix = np.empty(
        (len(reference) + 1) * (len(hypothesis) + 1), dtype=np.int32
    )
    operations = np.empty(max(1, len(reference) + len(hypothesis)), dtype=np.uint8)
    count = int(
        library().mji_trace(
            _address(ref),
            _address(hyp),
            len(reference),
            len(hypothesis),
            _address(matrix),
            _address(operations),
        )
    )
    return operations[:count][::-1].tolist()


def traces(
    references: list[list[int]], hypotheses: list[list[int]]
) -> list[list[int]]:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must contain the same number of items")
    count = len(references)
    if not count:
        return []
    ref, hyp, ref_offsets, hyp_offsets = _flatten_sequences(references, hypotheses)
    ref_lengths = np.diff(ref_offsets)
    hyp_lengths = np.diff(hyp_offsets)
    max_ref = int(ref_lengths.max(initial=0))
    max_hyp = int(hyp_lengths.max(initial=0))
    matrix_stride = (max_ref + 1) * (max_hyp + 1)
    cell_work = int(((ref_lengths + 1) * (hyp_lengths + 1)).sum())
    worker_count = min(16, count) if count >= 256 and cell_work >= 200_000 else 1
    worker_count = min(
        worker_count,
        max(1, (256 * 1024 * 1024) // (matrix_stride * np.dtype(np.int32).itemsize)),
    )
    matrices = np.empty(matrix_stride * worker_count, dtype=np.int32)
    operation_offsets = np.empty(count + 1, dtype=np.int64)
    operation_offsets[0] = 0
    np.cumsum(ref_lengths + hyp_lengths, out=operation_offsets[1:])
    operations = np.empty(max(1, int(operation_offsets[-1])), dtype=np.uint8)
    operation_counts = np.empty(count, dtype=np.int64)
    library().mji_traces(
        _address(ref),
        _address(hyp),
        _address(ref_offsets),
        _address(hyp_offsets),
        count,
        worker_count,
        _address(matrices),
        matrix_stride if worker_count > 1 else 0,
        _address(operation_offsets),
        _address(operations),
        _address(operation_counts),
    )
    return [
        operations[
            int(operation_offsets[index]) : int(operation_offsets[index])
            + int(operation_counts[index])
        ][::-1].tolist()
        for index in range(count)
    ]

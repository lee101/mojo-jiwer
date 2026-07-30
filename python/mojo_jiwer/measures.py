"""Scalar WER, MER, WIL, WIP, and CER measures."""

from __future__ import annotations

import numpy as np

from . import _native
from .process import _prepare, process_words
from .transformations import cer_default, wer_default


def _error_rate(reference, hypothesis, reference_transform, hypothesis_transform):
    _, _, references, hypotheses = _prepare(
        reference, hypothesis, reference_transform, hypothesis_transform
    )
    errors = sum(_native.distances(references, hypotheses))
    reference_length = sum(map(len, references))
    return errors / reference_length if reference_length else errors


def _default_character_error_rate(reference, hypothesis):
    references = [reference] if isinstance(reference, str) else reference
    hypotheses = [hypothesis] if isinstance(hypothesis, str) else hypothesis
    if (
        not isinstance(references, list)
        or not isinstance(hypotheses, list)
    ):
        return None
    try:
        references = [value.strip() for value in references] or [""]
        hypotheses = [value.strip() for value in hypotheses] or [""]
    except AttributeError:
        return None
    if len(references) != len(hypotheses):
        raise ValueError(
            "After applying the transforms on the reference and hypothesis "
            "sentences, their lengths must match. Instead got "
            f"{len(references)} reference and {len(hypotheses)} hypothesis sentences."
        )

    ref_offsets = np.empty(len(references) + 1, dtype=np.int64)
    hyp_offsets = np.empty(len(hypotheses) + 1, dtype=np.int64)
    ref_offsets[0] = hyp_offsets[0] = 0
    np.cumsum(
        np.fromiter(map(len, references), dtype=np.int64, count=len(references)),
        out=ref_offsets[1:],
    )
    np.cumsum(
        np.fromiter(map(len, hypotheses), dtype=np.int64, count=len(hypotheses)),
        out=hyp_offsets[1:],
    )

    ref_text = "".join(references)
    hyp_text = "".join(hypotheses)
    try:
        ref = np.frombuffer(
            ref_text.encode("utf-32le"), dtype="<u4"
        ).astype(np.int64)
        hyp = np.frombuffer(
            hyp_text.encode("utf-32le"), dtype="<u4"
        ).astype(np.int64)
    except UnicodeEncodeError:
        return None
    errors = sum(_native.distances_flat(ref, hyp, ref_offsets, hyp_offsets))
    reference_length = int(ref_offsets[-1])
    return errors / reference_length if reference_length else errors


def wer(
    reference=None,
    hypothesis=None,
    reference_transform=wer_default,
    hypothesis_transform=wer_default,
) -> float:
    return _error_rate(
        reference, hypothesis, reference_transform, hypothesis_transform
    )


def mer(
    reference=None,
    hypothesis=None,
    reference_transform=wer_default,
    hypothesis_transform=wer_default,
) -> float:
    return process_words(
        reference, hypothesis, reference_transform, hypothesis_transform
    ).mer


def wil(
    reference=None,
    hypothesis=None,
    reference_transform=wer_default,
    hypothesis_transform=wer_default,
) -> float:
    return process_words(
        reference, hypothesis, reference_transform, hypothesis_transform
    ).wil


def wip(
    reference=None,
    hypothesis=None,
    reference_transform=wer_default,
    hypothesis_transform=wer_default,
) -> float:
    return process_words(
        reference, hypothesis, reference_transform, hypothesis_transform
    ).wip


def cer(
    reference=None,
    hypothesis=None,
    reference_transform=cer_default,
    hypothesis_transform=cer_default,
) -> float:
    if (
        reference_transform is cer_default
        and hypothesis_transform is cer_default
    ):
        result = _default_character_error_rate(reference, hypothesis)
        if result is not None:
            return result
    return _error_rate(
        reference, hypothesis, reference_transform, hypothesis_transform
    )

"""jiwer-compatible processing results and alignment construction."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from . import _native
from .transformations import cer_default, wer_default

_MULTIPLE_SPACES = re.compile(r"\s\s+")


@dataclass
class AlignmentChunk:
    type: str
    ref_start_idx: int
    ref_end_idx: int
    hyp_start_idx: int
    hyp_end_idx: int

    def __post_init__(self):
        if self.type not in ["replace", "insert", "delete", "equal", "substitute"]:
            raise ValueError("")
        if self.type == "replace":
            self.type = "substitute"
        if self.ref_start_idx > self.ref_end_idx:
            raise ValueError(
                f"ref_start_idx={self.ref_start_idx} is larger than "
                f"ref_end_idx={self.ref_end_idx}"
            )
        if self.hyp_start_idx > self.hyp_end_idx:
            raise ValueError(
                f"hyp_start_idx={self.hyp_start_idx} is larger than "
                f"hyp_end_idx={self.hyp_end_idx}"
            )


@dataclass
class WordOutput:
    references: list[list[str]]
    hypotheses: list[list[str]]
    alignments: list[list[AlignmentChunk]]
    wer: float
    mer: float
    wil: float
    wip: float
    hits: int
    substitutions: int
    insertions: int
    deletions: int


@dataclass
class CharacterOutput:
    references: list[list[str]]
    hypotheses: list[list[str]]
    alignments: list[list[AlignmentChunk]]
    cer: float
    hits: int
    substitutions: int
    insertions: int
    deletions: int


def _apply_transform(sentence, transform, is_reference: bool):
    transformed = transform(sentence)
    if not isinstance(transformed, list) or any(
        not isinstance(part, list)
        or not all(isinstance(token, str) for token in part)
        for part in transformed
    ):
        side = "reference" if is_reference else "hypothesis"
        raise ValueError(
            "After applying the transformation, each "
            f"{side} should be a list of strings, with each string being a "
            "single word or character.Please ensure the given transformation "
            "reduces the input to a list of list strings."
        )
    return transformed


def _default_transform(sentence, transform, is_reference: bool):
    if not isinstance(sentence, (str, list)):
        return _apply_transform(sentence, transform, is_reference)
    values = [sentence] if isinstance(sentence, str) else sentence
    if not all(isinstance(value, str) for value in values):
        return _apply_transform(sentence, transform, is_reference)
    if transform is wer_default:
        if not values:
            return [[]]
        return [
            [
                word
                for word in _MULTIPLE_SPACES.sub(" ", value).strip().split(" ")
                if word
            ]
            for value in values
        ]
    if transform is cer_default:
        if not values:
            return [[]]
        return [list(value.strip()) for value in values]
    return _apply_transform(sentence, transform, is_reference)


def _encode(
    references: list[list[str]], hypotheses: list[list[str]]
) -> tuple[list[list[int]], list[list[int]]]:
    ids = defaultdict()
    ids.default_factory = ids.__len__
    return (
        [[ids[token] for token in sentence] for sentence in references],
        [[ids[token] for token in sentence] for sentence in hypotheses],
    )


def _chunks(operations: list[int]) -> list[AlignmentChunk]:
    if not operations:
        return []
    names = ("equal", "substitute", "delete", "insert")
    chunks: list[AlignmentChunk] = []
    ref_index = hyp_index = 0
    start = 0
    while start < len(operations):
        operation = operations[start]
        ref_start = ref_index
        hyp_start = hyp_index
        end = start
        while end < len(operations) and operations[end] == operation:
            if operation != 3:
                ref_index += 1
            if operation != 2:
                hyp_index += 1
            end += 1
        chunks.append(
            AlignmentChunk(
                names[operation], ref_start, ref_index, hyp_start, hyp_index
            )
        )
        start = end
    return chunks


def _prepare(reference, hypothesis, reference_transform, hypothesis_transform):
    if isinstance(reference, str):
        reference = [reference]
    if isinstance(hypothesis, str):
        hypothesis = [hypothesis]
    references = _default_transform(reference, reference_transform, True)
    hypotheses = _default_transform(hypothesis, hypothesis_transform, False)
    if len(references) != len(hypotheses):
        raise ValueError(
            "After applying the transforms on the reference and hypothesis "
            "sentences, their lengths must match. Instead got "
            f"{len(references)} reference and {len(hypotheses)} hypothesis sentences."
        )
    ref_ids, hyp_ids = _encode(references, hypotheses)
    return references, hypotheses, ref_ids, hyp_ids


def process_words(
    reference,
    hypothesis,
    reference_transform=wer_default,
    hypothesis_transform=wer_default,
) -> WordOutput:
    references, hypotheses, ref_ids, hyp_ids = _prepare(
        reference, hypothesis, reference_transform, hypothesis_transform
    )
    alignments = []
    hits = substitutions = deletions = insertions = 0
    reference_words = hypothesis_words = 0

    for ref, hyp, operations in zip(
        ref_ids, hyp_ids, _native.traces(ref_ids, hyp_ids)
    ):
        sentence_chunks = _chunks(operations)
        for chunk in sentence_chunks:
            if chunk.type == "equal":
                hits += chunk.ref_end_idx - chunk.ref_start_idx
            elif chunk.type == "substitute":
                substitutions += chunk.ref_end_idx - chunk.ref_start_idx
            elif chunk.type == "delete":
                deletions += chunk.ref_end_idx - chunk.ref_start_idx
            else:
                insertions += chunk.hyp_end_idx - chunk.hyp_start_idx
        reference_words += len(ref)
        hypothesis_words += len(hyp)
        alignments.append(sentence_chunks)

    if reference_words == 0:
        word_error_rate = insertions
        if hypothesis_words == 0:
            match_error_rate = 0
            word_information_preserved = 1
        else:
            match_error_rate = 1
            word_information_preserved = 0
    else:
        errors = substitutions + deletions + insertions
        word_error_rate = errors / (hits + substitutions + deletions)
        match_error_rate = errors / (hits + substitutions + deletions + insertions)
        word_information_preserved = (
            (hits / reference_words) * (hits / hypothesis_words)
            if hypothesis_words
            else 0
        )

    return WordOutput(
        references,
        hypotheses,
        alignments,
        word_error_rate,
        match_error_rate,
        1 - word_information_preserved,
        word_information_preserved,
        hits,
        substitutions,
        insertions,
        deletions,
    )


def process_characters(
    reference,
    hypothesis,
    reference_transform=cer_default,
    hypothesis_transform=cer_default,
) -> CharacterOutput:
    result = process_words(
        reference, hypothesis, reference_transform, hypothesis_transform
    )
    return CharacterOutput(
        result.references,
        result.hypotheses,
        result.alignments,
        result.wer,
        result.hits,
        result.substitutions,
        result.insertions,
        result.deletions,
    )


def collect_error_counts(output: WordOutput | CharacterOutput):
    substitutions = defaultdict(int)
    insertions = defaultdict(int)
    deletions = defaultdict(int)
    for index, chunks in enumerate(output.alignments):
        reference = output.references[index]
        hypothesis = output.hypotheses[index]
        separator = " " if isinstance(output, WordOutput) else ""
        for chunk in chunks:
            if chunk.type == "insert":
                insertions[
                    separator.join(
                        hypothesis[chunk.hyp_start_idx : chunk.hyp_end_idx]
                    )
                ] += 1
            elif chunk.type == "delete":
                deletions[
                    separator.join(reference[chunk.ref_start_idx : chunk.ref_end_idx])
                ] += 1
            elif chunk.type == "substitute":
                replaced = separator.join(
                    reference[chunk.ref_start_idx : chunk.ref_end_idx]
                )
                replacement = separator.join(
                    hypothesis[chunk.hyp_start_idx : chunk.hyp_end_idx]
                )
                substitutions[(replaced, replacement)] += 1
    return substitutions, insertions, deletions

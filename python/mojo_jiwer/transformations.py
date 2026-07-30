"""Default transformation pipelines."""

from .transforms import (
    Compose,
    ReduceToListOfListOfChars,
    ReduceToListOfListOfWords,
    ReduceToSingleSentence,
    RemoveMultipleSpaces,
    Strip,
)

wer_default = Compose(
    [RemoveMultipleSpaces(), Strip(), ReduceToListOfListOfWords()]
)
cer_default = Compose([Strip(), ReduceToListOfListOfChars()])
wer_contiguous = Compose(
    [
        RemoveMultipleSpaces(),
        Strip(),
        ReduceToSingleSentence(),
        ReduceToListOfListOfWords(),
    ]
)
cer_contiguous = Compose(
    [Strip(), ReduceToSingleSentence(), ReduceToListOfListOfChars()]
)

__all__ = ["wer_default", "cer_default", "wer_contiguous", "cer_contiguous"]

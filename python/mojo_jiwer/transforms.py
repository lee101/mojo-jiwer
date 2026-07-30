"""The core jiwer transforms needed by WER and CER."""

from __future__ import annotations

import re


class AbstractTransform:
    def __call__(self, sentences: str | list[str]):
        if isinstance(sentences, str):
            return self.process_string(sentences)
        if isinstance(sentences, list):
            return self.process_list(sentences)
        raise ValueError(
            f"input {sentences} was expected to be a string or list of strings"
        )

    def process_string(self, value: str):
        raise NotImplementedError

    def process_list(self, values: list[str]):
        return [self.process_string(value) for value in values]


class Compose:
    def __init__(self, transforms: list[AbstractTransform]):
        self.transforms = transforms

    def __call__(self, text):
        for transform in self.transforms:
            text = transform(text)
        return text


class RemoveMultipleSpaces(AbstractTransform):
    def process_string(self, value: str):
        return re.sub(r"\s\s+", " ", value)


class Strip(AbstractTransform):
    def process_string(self, value: str):
        return value.strip()


class ToLowerCase(AbstractTransform):
    def process_string(self, value: str):
        return value.lower()


class ToUpperCase(AbstractTransform):
    def process_string(self, value: str):
        return value.upper()


class RemovePunctuation(AbstractTransform):
    def process_string(self, value: str):
        import unicodedata

        return "".join(
            character
            for character in value
            if not unicodedata.category(character).startswith("P")
        )


class ReduceToListOfListOfWords(AbstractTransform):
    def __init__(self, word_delimiter: str = " "):
        self.word_delimiter = word_delimiter

    def process_string(self, value: str):
        return [[word for word in value.split(self.word_delimiter) if word]]

    def process_list(self, values: list[str]):
        if not values:
            return [[]]
        return [self.process_string(value)[0] for value in values]


class ReduceToListOfListOfChars(AbstractTransform):
    def process_string(self, value: str):
        return [list(value)]

    def process_list(self, values: list[str]):
        if not values:
            return [[]]
        return [self.process_string(value)[0] for value in values]


class ReduceToSingleSentence(AbstractTransform):
    def __init__(self, word_delimiter: str = " "):
        self.word_delimiter = word_delimiter

    def process_string(self, value: str):
        return value

    def process_list(self, values: list[str]):
        filtered = [value for value in values if value]
        return [self.word_delimiter.join(filtered)] if filtered else []

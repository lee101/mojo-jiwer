"""Transform-level parity: every covered transform is compared with upstream jiwer."""

from __future__ import annotations

import pytest

upstream = pytest.importorskip("jiwer")

import mojo_jiwer as mj

TRANSFORM_NAMES = [
    "Compose",
    "ReduceToListOfListOfChars",
    "ReduceToListOfListOfWords",
    "ReduceToSingleSentence",
    "RemoveMultipleSpaces",
    "RemovePunctuation",
    "Strip",
    "ToLowerCase",
    "ToUpperCase",
]

SENTENCES = [
    "",
    " ",
    "a",
    "  a   b  ",
    "Hello, WORLD!",
    "¿Qué tal? ¡Bien!",
    "naïve café",
    "你好世界",
    "e\u0301",
    "😀 emoji",
    "\t\n",
    "a\u00a0b",
]

LISTS = [
    [],
    [""],
    ["a"],
    ["a", "b b"],
    ["", "a", ""],
    ["Hello, WORLD!", "Unicode: ¿Qué tal?"],
]


def test_transform_names_are_exported_by_upstream():
    for name in TRANSFORM_NAMES:
        assert hasattr(upstream, name), f"upstream jiwer has no {name}"
        assert hasattr(mj, name)


@pytest.mark.parametrize(
    "name", [n for n in TRANSFORM_NAMES if n != "Compose"]
)
@pytest.mark.parametrize("argument", SENTENCES + LISTS)
def test_transform_output_matches_upstream(name, argument):
    assert getattr(mj, name)()(argument) == getattr(upstream, name)()(argument)


@pytest.mark.parametrize("delimiter", [" ", "", "x", "-"])
def test_reduce_word_delimiter_matches_upstream(delimiter):
    reference = ["a b", "a", "a  b", ""]

    def outcome(module, name):
        try:
            return (True, getattr(module, name)(delimiter)(reference))
        except ValueError as exc:
            return (False, str(exc))

    for name in ["ReduceToListOfListOfWords", "ReduceToSingleSentence"]:
        ours = outcome(mj, name)
        theirs = outcome(upstream, name)
        if theirs[0]:
            assert ours == theirs
        else:
            # upstream transforms.py: str.split("") raises "empty separator".
            assert "empty separator" in theirs[1]
            assert ours == (False, theirs[1])


@pytest.mark.parametrize(
    "argument", [None, 3, 2.5, b"abc", ("a",), {"a": "b"}, [None], [1]]
)
def test_transform_rejects_non_text_like_upstream(argument):
    # upstream transforms.py AbstractTransform.__call__ (and process_list) raise
    # ValueError/TypeError/AttributeError; the class must match, not just "an error".
    for name in TRANSFORM_NAMES:
        if name == "Compose":
            continue
        ours = getattr(mj, name)()
        theirs = getattr(upstream, name)()
        try:
            expected = theirs(argument)
        except Exception as exc:  # noqa: BLE001
            with pytest.raises(type(exc)):
                ours(argument)
        else:
            assert ours(argument) == expected


def test_remove_single_element_none_lists_raise_same_class_as_upstream():
    # upstream transforms.py ReduceToSingleSentence.process_list calls len(i) ->
    # TypeError; AbstractTransform.process_list is fine, but the filter is not.
    with pytest.raises(TypeError):
        upstream.ReduceToSingleSentence()(["a", None])
    with pytest.raises(TypeError):
        mj.ReduceToSingleSentence()(["a", None])


def test_remove_punctuation_none_element_raises_same_class_as_upstream():
    # upstream transforms.py BaseRemoveTransform.process_string uses str.replace
    # on the element -> AttributeError for None.
    with pytest.raises(AttributeError):
        upstream.RemovePunctuation()(["a", None])
    with pytest.raises(AttributeError):
        mj.RemovePunctuation()(["a", None])


def test_compose_matches_upstream_pipeline_output():
    def build(module):
        return module.Compose(
            [
                module.RemoveMultipleSpaces(),
                module.Strip(),
                module.ToLowerCase(),
                module.RemovePunctuation(),
                module.ReduceToListOfListOfWords(),
            ]
        )

    ours = build(mj)
    theirs = build(upstream)
    for argument in SENTENCES + LISTS:
        assert ours(argument) == theirs(argument)


def test_default_and_contiguous_pipelines_match_upstream():
    for name in ["wer_default", "cer_default", "wer_contiguous", "cer_contiguous"]:
        ours = getattr(mj, name)
        theirs = getattr(upstream, name)
        for argument in SENTENCES + LISTS:
            assert ours(argument) == theirs(argument), f"{name}({argument!r})"


@pytest.mark.parametrize("name", ["wer_default", "cer_default", "wer_contiguous", "cer_contiguous"])
def test_pipelines_accept_the_other_modules_transform_objects(name):
    # Custom pipelines mix transforms; upstream accepts its own objects, so does ours.
    ours = getattr(mj, name)
    theirs = getattr(upstream, name)
    assert ours(["a b", "c"]) == theirs(["a b", "c"])


def test_abstract_transform_rejects_non_text_like_upstream():
    class Ours(mj.AbstractTransform):
        def process_string(self, value):
            return value

    class Theirs(upstream.AbstractTransform):
        def process_string(self, value):
            return value

    for argument in [None, 3, ("a",)]:
        with pytest.raises(ValueError, match="expected to be a string or list"):
            Ours()(argument)
        with pytest.raises(ValueError, match="expected to be a string or list"):
            Theirs()(argument)

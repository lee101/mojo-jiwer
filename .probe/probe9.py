import sys
from dataclasses import asdict

sys.path.insert(0, "/nvme0n1-disk/code/mojo-jiwer/python")
import jiwer as up

# Duplicate/boundary fixtures for tie-breaking (encoded upstream values)
for ref, hyp in [
    ("a b c", "b c d"),
    ("a b c d", "a x y d"),
    ("the quick brown fox", "the brown quick fox"),
    ("a b", "b a"),
    ("a b", "b b b a"),
    ("a", "a a"),
]:
    out = up.process_words(ref, hyp)
    print(repr(ref), repr(hyp))
    print("   counts", out.hits, out.substitutions, out.insertions, out.deletions)
    print("   align", [(c.type, c.ref_start_idx, c.ref_end_idx, c.hyp_start_idx, c.hyp_end_idx) for c in out.alignments[0]])

# multiword boundary fixtures (63/64/127/257 words around the 64-bit word split)
for length in [63, 64, 65, 127, 257]:
    reference = [index % 29 for index in range(length)]
    hypothesis = reference.copy()
    for index in range(3, length, 37):
        hypothesis[index] = 100 + index % 11
    hypothesis.insert(length // 3, 500)
    del hypothesis[(2 * length) // 3]
    words_r = " ".join(f"w{t}" for t in reference)
    words_h = " ".join(f"w{t}" for t in hypothesis)
    out = up.process_words(words_r, words_h)
    print("boundary", length, "wer", out.wer, "counts", out.hits, out.substitutions, out.insertions, out.deletions)

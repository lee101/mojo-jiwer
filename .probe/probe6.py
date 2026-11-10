import random
import sys
from dataclasses import asdict

sys.path.insert(0, "/nvme0n1-disk/code/mojo-jiwer/python")
import jiwer as up
import mojo_jiwer as mj

rng = random.Random(3)
for n, m, vocab in [(70, 70, 2), (200, 200, 2), (500, 500, 2), (500, 500, 20), (1000, 1000, 1)]:
    r = " ".join(str(rng.randrange(vocab)) for _ in range(n))
    h = " ".join(str(rng.randrange(vocab)) for _ in range(m))
    a = asdict(mj.process_words(r, h))
    b = asdict(up.process_words(r, h))
    print(n, m, vocab, "match" if a == b else "MISMATCH", a["wer"], b["wer"])

# repeated token only
for n, m in [(300, 300), (100, 400), (400, 100)]:
    r = " ".join(["a"] * n)
    h = " ".join(["b"] * m)
    a = asdict(mj.process_words(r, h))
    b = asdict(up.process_words(r, h))
    print("ab", n, m, "match" if a == b else "MISMATCH")

"""Isolate the integer-overflow panic: public API vs hand-built kernel call."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import bench as b
import mojo_jiwer as mojo
from mojo_jiwer import _native
from mojo_jiwer.process import _default_transform, _encode
from mojo_jiwer.transformations import wer_default

refs, hyps = b.word_corpus(10_000, 12, 1)
print("public wer:", mojo.wer(refs, hyps), flush=True)

r_tok = _default_transform(refs, wer_default, True)
h_tok = _default_transform(hyps, wer_default, False)
a = _encode(r_tok, h_tok)
print("public distances:", sum(_native.distances(a[0], a[1])), flush=True)
print("public traces count:", len(_native.traces(a[0], a[1])), flush=True)

# Hand-built: reproduce distances_flat's exact buffer layout at worker_count=1
pairs = [(x, y) if len(y) <= len(x) else (y, x) for x, y in zip(a[0], a[1])]
import time
t = time.perf_counter()
print("via distances_flat:", sum(_native.distances_flat(*_native._flatten_sequences([p[0] for p in pairs], [p[1] for p in pairs]))),
      f"{(time.perf_counter()-t)*1e3:.2f} ms", flush=True)

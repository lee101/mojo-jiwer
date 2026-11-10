import random
# Python semantics: start end end defaults(0, len); negative += len; clamp to [0, len].
def two_sweeps(s):
    start = 0
    while start < len(s) and s[start].isspace(): start += 1
    end = len(s)
    while end > start and s[end-1].isspace(): end -= 1
    return s[start:end]
def set_strip(s, chars):
    start, end = 0, len(s)
    while start < end and s[start] in chars: start += 1
    while end > start and s[end-1] in chars: end -= 1
    return s[start:end]
chars={chr(c) for c in range(0x110000) if chr(c).isspace() and len(chr(c).strip(chr(c)))==0}
chars={" ","\t","\n","\r","\x0b","\x0c"}
# exhaustive over a small alphabet of whitespace-like and non-space chars
alpha=[" ","\t","\n","\r","\x0b","\x0c","\u00a0","\u3000","\u2028","a","é"]
bad=0
rng=random.Random(0)
for _ in range(200000):
    s="".join(rng.choice(alpha) for _ in range(rng.randrange(0,9)))
    if two_sweeps(s)!=set_strip(s,chars):
        bad+=1
        if bad<=5: print(repr(s), repr(two_sweeps(s)), repr(set_strip(s,chars)))
print("mismatches:",bad)

import random
W = 8
def scan(left, right, limit):
    i = 0
    while i + W <= limit:
        if not all(left[i+k] == right[i+k] for k in range(W)):   # break: scalar tail
            break
        i += W
    while i < limit and left[i] == right[i]:
        i += 1
    return i
def truth(left, right):
    i = 0
    while i < min(len(left), len(right)) and left[i] == right[i]: i += 1
    return i
rng = random.Random(0)
bad = 0
for trial in range(20000):
    n = rng.randrange(0, 40)
    left = [rng.randrange(4) for _ in range(n)]
    right = list(left)
    for _ in range(rng.randrange(1, 4)):          # a cluster of mismatches
        if n: right[rng.randrange(n)] = rng.randrange(4, 8)
    got = scan(left, right, min(n, len(right)))
    want = truth(left, right)
    if got != want:
        bad += 1
        if bad <= 5:
            mismatch = [i for i in range(n) if left[i] != right[i]]
            print(f"n={n} mismatches at {mismatch}: scan gives {got}, exact answer {want}")
print(f"{bad} of 20000 random cases")

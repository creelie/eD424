#!/usr/bin/env python3
"""
Broaden the search: find A_18's TRUE stabilizer within the FULL signed-
permutation automorphism group of the D4 root system (order 384 = 2^4*4!),
not just the 48-element subgroup fixing coordinate 0's identity+sign
(that was based on an incorrect guess about A_18's coordinate description).
"""
import numpy as np
import itertools

def build_roots():
    roots = []
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(4)
                    v[i] = si; v[j] = sj
                    roots.append(v / np.sqrt(2))
    return np.array(roots)

roots = build_roots()
roots_int = np.round(roots*np.sqrt(2)).astype(int)
root_to_idx = {tuple(r): i for i, r in enumerate(roots_int)}
A_18 = {0,2,4,5,6,7,8,9,10,11,12,13,16,17,20,21,22,23}

group_elems = []
for perm in itertools.permutations(range(4)):
    for signs in itertools.product([1,-1], repeat=4):
        M = np.zeros((4,4), dtype=int)
        for out_idx in range(4):
            M[out_idx, perm[out_idx]] = signs[out_idx]
        group_elems.append(M)
print(f"Full signed-permutation group size: {len(group_elems)} (expect 384)")

stabilizer = []
for M in group_elems:
    image_idxs = {}
    ok = True
    for i, r in enumerate(roots_int):
        rimg = tuple((M @ r))
        if rimg not in root_to_idx:
            ok = False
            break
        image_idxs[i] = root_to_idx[rimg]
    if not ok:
        continue
    image_A18 = set(image_idxs[i] for i in A_18)
    if image_A18 == A_18:
        stabilizer.append((M, image_idxs))

print(f"TRUE stabilizer of A_18 within the full 384-element group: {len(stabilizer)} elements")
print()
print("Sample of a few stabilizer matrices:")
for M, _ in stabilizer[:6]:
    print(M.tolist())

# Non-identity element: describe it as (perm, signs)
def describe(M):
    perm = [int(np.nonzero(M[i])[0][0]) for i in range(4)]
    signs = [int(M[i, perm[i]]) for i in range(4)]
    return perm, signs

print()
print("Describing all stabilizer elements as (perm, signs):")
for M, _ in stabilizer:
    p, s = describe(M)
    print(f"  perm={p}, signs={s}")

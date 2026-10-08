import numpy as np
from itertools import product, combinations

# D4 roots in the {+-e_i +- e_j} realization
roots = []
for i,j in combinations(range(4),2):
    for si in (1,-1):
        for sj in (1,-1):
            v = np.zeros(4)
            v[i] = si; v[j] = sj
            roots.append(v)
roots = np.array(roots)  # 24 x 4, each norm sqrt(2)

def root_set_key(R, tol=1e-6):
    # canonical set of rounded tuples for membership testing
    return set(tuple(np.round(r, 6)) for r in R)

base_key = root_set_key(roots)

def preserves_root_set(g, tol=1e-6):
    img = roots @ g.T
    key = root_set_key(img)
    return key == base_key

# Candidate: normalized Hadamard matrix (orthogonal)
H = np.array([[1,1,1,1],
              [1,1,-1,-1],
              [1,-1,1,-1],
              [1,-1,-1,1]], dtype=float) / 2.0
print("H orthogonal check (H H^T):")
print(np.round(H @ H.T, 6))
print("H preserves 24-root set?", preserves_root_set(H))

# Try all 24 sign-flip / coordinate-permutation-conjugates of H to see if any preserves the set,
# in case a specific orientation/signing of the Hadamard matrix is needed.
from itertools import permutations
found = []
perms = list(permutations(range(4)))
signs = list(product([1,-1], repeat=4))
count = 0
for perm in perms:
    P = np.eye(4)[list(perm)]
    for s in signs:
        S = np.diag(s)
        g = S @ P @ H
        count += 1
        if preserves_root_set(g):
            found.append((perm, s))
print(f"Tested {count} sign/permutation-conjugates of H; preserved root set in {len(found)} cases")
if found:
    perm, s = found[0]
    P = np.eye(4)[list(perm)]
    S = np.diag(s)
    g = S @ P @ H
    print("Example g:\n", np.round(g,4))
    print("g^3 =\n", np.round(g@g@g,4))
    print("det(g) =", np.round(np.linalg.det(g),4))

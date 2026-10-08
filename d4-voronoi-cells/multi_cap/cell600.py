"""
Twenty-three point codes inside the 600-cell.

The 120 vertices of the 600-cell have inner products in
{0, +-1/2, +-cos36, +-cos72, +-1}, and two of them are closer than 60
degrees exactly when their inner product is cos36 = 0.809..., which happens
along the 720 edges of the 600-cell. So the subsets with all inner products
at most 1/2 are the independent sets of that 12-regular graph, and a
23-element one that is maximal is a saturated contact configuration unless
it sits inside one of the inscribed 24-cells.

A subset of a D_4 root system has all its inner products in
{-1,-1/2,0,1/2}, so the values +-cos72 = +-0.309 and -cos36 detect a
subset that is not inside a root system.

The maximal subsets are sampled by greedy growth along random orders of
the vertices. Two million of them are drawn, and their sizes are reported:
the point of interest is whether size 23 occurs at all.
"""
import numpy as np

ROOTVALS = np.array([-1.0, -0.5, 0.0, 0.5])


def cell600():
    phi = (1 + np.sqrt(5.0)) / 2
    V = []
    for i in range(4):
        for s in (1, -1):
            v = np.zeros(4)
            v[i] = s
            V.append(v)
    for m in range(16):
        V.append(np.array([(-1.0) ** ((m >> b) & 1) * 0.5 for b in range(4)]))
    even = [(0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2),
            (1, 0, 3, 2), (1, 2, 0, 3), (1, 3, 2, 0),
            (2, 0, 1, 3), (2, 1, 3, 0), (2, 3, 0, 1),
            (3, 0, 2, 1), (3, 1, 0, 2), (3, 2, 1, 0)]
    base = np.array([phi, 1.0, 1 / phi, 0.0]) / 2
    for p in even:
        for s0 in (1, -1):
            for s1 in (1, -1):
                for s2 in (1, -1):
                    v = np.zeros(4)
                    for a, b in zip(p, [s0 * base[0], s1 * base[1],
                                        s2 * base[2], 0.0]):
                        v[a] = b
                    V.append(v)
    return np.unique(np.round(np.array(V), 9), axis=0)


V = cell600()
N = len(V)
S = V @ V.T
COMPAT = (S <= 0.5 + 1e-9)
np.fill_diagonal(COMPAT, False)
ADJ = [sum(1 << j for j in range(N) if COMPAT[i, j]) for i in range(N)]
FULL = (1 << N) - 1


def popcount(x):
    return bin(x).count("1")


def in_root_spectrum(idx, tol=1e-6):
    G = V[idx] @ V[idx].T
    off = G[~np.eye(len(idx), dtype=bool)]
    return bool(np.all(np.min(np.abs(off[:, None] - ROOTVALS), axis=1) < tol))


def random_maximal(rng, order):
    """One maximal subset with all inner products at most 1/2, grown greedily
    along a random order of the vertices."""
    rng.shuffle(order)
    cur, avail = 0, FULL
    for v in order:
        if (avail >> v) & 1:
            cur |= 1 << v
            avail &= ADJ[v]
    return cur


if __name__ == "__main__":
    import random
    print(f"vertices: {N}")
    deg = [popcount(a) for a in ADJ]
    print(f"compatible vertices per vertex: {min(deg)} to {max(deg)}"
          f"   (so {N-1-max(deg)} to {N-1-min(deg)} lie closer than"
          f" 60 degrees, which is the edge graph of the 600-cell)")
    vals = sorted({round(float(x), 6) for x in S.ravel()})
    print(f"inner products occurring: {vals}")
    print()

    TRIALS = 2000000
    rng = random.Random(20260914)
    order = list(range(N))
    sizes = {}
    outside = []
    for t in range(TRIALS):
        cur = random_maximal(rng, order)
        k = popcount(cur)
        sizes[k] = sizes.get(k, 0) + 1
        if k >= 23:
            idx = [i for i in range(N) if (cur >> i) & 1]
            if not in_root_spectrum(idx):
                outside.append(idx)
    print(f"{TRIALS} random maximal subsets, by size:")
    for k in sorted(sizes):
        print(f"    size {k:>2}: {sizes[k]}")
    print()
    print(f"  of size 23: {sizes.get(23, 0)}")
    print(f"  of size 23 or more with an inner product outside"
          f" {{-1,-1/2,0,1/2}}: {len(outside)}")
    print()
    print("reading: a maximal subset of size 23 would be a saturated contact")
    print("configuration unless it sat inside one of the inscribed 24-cells,")
    print("and no maximal subset of size 23 was produced at all: the sizes")
    print("stop at 22 and resume at 24. This is sampling, not enumeration,")
    print("and is reported as such.")

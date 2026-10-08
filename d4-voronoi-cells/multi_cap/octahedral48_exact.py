#!/usr/bin/env python3
"""
octahedral48_exact.py -- the 48 unit quaternions of the binary octahedral
group, which are the vertices of a compound of two 24-cells (the Hurwitz
units, a copy of the D4 root system, and their images under multiplication
by (1+i)/sqrt2, a second copy in dual position).  Two of the 48 are closer
than 60 degrees exactly when they lie in different copies and their inner
product is 1/sqrt2.  This script enumerates, exactly, every subset of size
23 and 24 with all pairwise inner products at most 1/2, and finds only the
two copies of the root system and their 48 one-point deletions.

Coordinates are doubled and written in Z[sqrt2] as pairs (a, b) standing
for a + b*sqrt2, so every inner product is exact.  Runtime a second.
"""
import itertools

def zmul(x, y):
    a, b = x; c, d = y
    return (a*c + 2*b*d, a*d + b*c)
def zadd(x, y): return (x[0]+y[0], x[1]+y[1])
def inner4(u, v):
    s = (0, 0)
    for a, b in zip(u, v): s = zadd(s, zmul(a, b))
    return s

def build():
    V = []
    for i in range(4):
        for s in (1, -1):
            v = [(0,0)]*4; v[i] = (2*s, 0); V.append(tuple(v))
    for signs in itertools.product((1, -1), repeat=4):
        V.append(tuple((s, 0) for s in signs))
    for i in range(4):
        for j in range(i+1, 4):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [(0,0)]*4; v[i] = (0, si); v[j] = (0, sj); V.append(tuple(v))
    assert len(V) == 48
    return V

def main():
    V = build(); n = len(V)
    for v in V: assert inner4(v, v) == (4, 0)
    allowed = {(4,0), (-4,0), (2,0), (-2,0), (0,0), (0,2), (0,-2)}   # 1, -1, 1/2, -1/2, 0, 1/sqrt2, -1/sqrt2
    adj = [0]*n; cadj = [0]*n
    for i in range(n):
        for j in range(i+1, n):
            ip = inner4(V[i], V[j]); assert ip in allowed, ip
            if ip == (0, 2):
                adj[i] |= 1 << j; adj[j] |= 1 << i
            if ip in {(-4,0), (-2,0), (0,0), (2,0)}:
                cadj[i] |= 1 << j; cadj[j] |= 1 << i
    degs = {bin(a).count('1') for a in adj}
    print(f"48 vertices; pairs at 45 degrees form a graph with degrees {sorted(degs)}")
    # enumerate independent sets of size >= 23 by simple recursion (48 vertices)
    found = {23: [], 24: [], 25: []}
    def rec(chosen, cand, start, size):
        if size >= 23: found[min(size, 25)].append(chosen)
        if size >= 25: return
        c = cand
        while c:
            v = (c & -c).bit_length() - 1; c &= ~(1 << v)
            rec(chosen | (1 << v), c & ~adj[v], v+1, size+1)
    rec(0, (1 << n) - 1, 0, 0)
    n24 = len(found[24]); n23 = len(found[23]); n25 = len(found[25])
    print(f"independent sets: size 25: {n25}, size 24: {n24}, size 23: {n23}")
    cells = found[24]
    for c in cells:
        mem = [i for i in range(n) if c >> i & 1]
        assert all(inner4(V[i], V[j]) in {(-4,0), (-2,0), (0,0), (2,0)} for i, j in itertools.combinations(mem, 2))
    outside = [c for c in found[23] if not any((c & ~cell) == 0 for cell in cells)]
    print(f"the {n24} sets of size 24 are copies of the root system; of the {n23} sets of size 23, "
          f"{len(outside)} lie outside every copy")
    assert n25 == 0 and n24 == 2 and n23 == 48 and not outside
    print("PASS: among the 48 vertices of the two dual 24-cells, every 23-point code of minimal "
          "angle 60 degrees is one of the two root systems less a vertex.")

if __name__ == "__main__":
    main()

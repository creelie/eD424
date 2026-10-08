#!/usr/bin/env python3
"""
cell600_exact.py -- every 23-point code of minimal angle 60 degrees among
the vertices of the 600-cell lies in an inscribed 24-cell.

The 120 vertices of the 600-cell are the unit icosians.  Two of them are
closer than 60 degrees exactly when their inner product is cos 36 = phi/2,
that is, when they are joined by an edge; so the subsets whose pairwise
inner products are all at most 1/2 are exactly the independent sets of the
edge graph, which is 12-regular on 120 vertices.  cell600.py samples
maximal independent sets at random.  This script settles the question
exhaustively.

  (1) Exact construction.  The vertices are built in Z[phi], with every
      coordinate doubled so that each is one of 0, +-1, +-2, +-phi,
      +-(phi-1), and inner products are computed in Z[phi] with no
      floating point.  The nine values 4<u,v> in {0, +-2, +-4, +-2phi,
      +-(2phi-2)} are checked to be the only ones that occur.

  (2) The inscribed 24-cells.  A subset all of whose inner products lie in
      {-1, -1/2, 0, 1/2} is a clique of a second graph on the same
      vertices; the 24-cliques of that graph are found by exhaustive
      backtracking.  There are 25, each vertex lies in 5 of them, and each
      is an independent set of the edge graph.

  (3) The independence number is 24.  No independent set of size 25
      exists: backtracking with a clique-cover bound over all sets
      containing a fixed vertex returns nothing.

  (4) Every independent set of size 23 lies in one of the 25 cells.  The
      independent 23-sets containing a fixed vertex are enumerated
      completely; there are 115 of them, so 115 x 120 / 23 = 600 in all,
      which is 25 x 24, one for each (cell, deleted vertex), and each of
      the 115 lies in a cell.

Step (4) is repeated by cell600_enum.c, an independent bitset
implementation in C reading the graph from cell600_graph.txt, which this
script writes.  The SAT check of the same statement (python-sat, CaDiCaL)
is also available with --sat; it is slower on this symmetric instance and
is not needed for the result.
"""
import sys, itertools, time

# ---------- exact arithmetic in Z[phi]: pair (a,b) means a + b*phi, phi^2 = phi + 1
def zmul(x, y):
    a, b = x; c, d = y
    return (a*c + b*d, a*d + b*c + b*d)
def zadd(x, y):
    return (x[0]+y[0], x[1]+y[1])

def build_vertices():
    V = set()
    for i in range(4):
        for s in (1, -1):
            v = [(0,0)]*4; v[i] = (2*s, 0); V.add(tuple(v))
    for signs in itertools.product((1,-1), repeat=4):
        V.add(tuple((s,0) for s in signs))
    base = [(0,1), (1,0), (-1,1), (0,0)]        # phi, 1, phi-1, 0   (all doubled)
    def is_even(p):
        return sum(1 for i in range(4) for j in range(i+1,4) if p[i] > p[j]) % 2 == 0
    for p in itertools.permutations(range(4)):
        if not is_even(p): continue
        for signs in itertools.product((1,-1), repeat=3):
            v = [None]*4
            for slot, idx in enumerate(p):
                val = base[slot]
                if slot < 3:
                    val = (val[0]*signs[slot], val[1]*signs[slot])
                v[idx] = val
            V.add(tuple(v))
    V = sorted(V)
    assert len(V) == 120, len(V)
    return V

def inner4(u, v):
    s = (0,0)
    for a, b in zip(u, v):
        s = zadd(s, zmul(a, b))
    return s

def popcount(x): return bin(x).count('1')

def cover_bound(cand, adj):
    k = 0
    while cand:
        k += 1
        v = (cand & -cand).bit_length() - 1
        cand &= ~(1 << v)
        cc = cand & adj[v]
        while cc:
            u = (cc & -cc).bit_length() - 1
            cand &= ~(1 << u)
            cc &= adj[u]
    return k

def enumerate_independent(adj, n, size, fixed=0):
    """All independent sets of the given size containing vertex `fixed`."""
    out = []
    def rec(chosen, cand, need):
        if need == 0:
            out.append(chosen); return
        if popcount(cand) < need: return
        if cover_bound(cand, adj) < need: return
        while cand:
            if popcount(cand) < need: return
            v = (cand & -cand).bit_length() - 1
            cand &= ~(1 << v)
            rec(chosen | (1 << v), cand & ~adj[v], need - 1)
    full = (1 << n) - 1
    cand = (full & ~(1 << fixed)) & ~adj[fixed]
    rec(1 << fixed, cand, size - 1)
    return out

def enumerate_cliques(cadj, n, size, fixed=0):
    """All cliques of the given size containing `fixed` in the graph cadj."""
    out = []
    def rec(chosen, cand, need):
        if need == 0:
            out.append(chosen); return
        if popcount(cand) < need: return
        while cand:
            if popcount(cand) < need: return
            v = (cand & -cand).bit_length() - 1
            cand &= ~(1 << v)
            rec(chosen | (1 << v), cand & cadj[v], need - 1)
    rec(1 << fixed, cadj[fixed], size - 1)
    return out

def main():
    t0 = time.time()
    V = build_vertices(); n = len(V)
    for v in V: assert inner4(v, v) == (4, 0)
    allowed = {(4,0), (-4,0), (0,2), (0,-2), (2,0), (-2,0), (-2,2), (2,-2), (0,0)}
    rootlike = {(-4,0), (-2,0), (0,0), (2,0)}          # <u,v> in {-1,-1/2,0,1/2}
    adj = [0]*n; cadj = [0]*n; edges = []
    for i in range(n):
        for j in range(i+1, n):
            ip = inner4(V[i], V[j])
            assert ip in allowed, ip
            if ip == (0, 2):
                edges.append((i, j)); adj[i] |= 1 << j; adj[j] |= 1 << i
            if ip in rootlike:
                cadj[i] |= 1 << j; cadj[j] |= 1 << i
    assert all(popcount(a) == 12 for a in adj)
    assert all(popcount(a) == 71 for a in cadj)        # 1 + 6... : -1 (1), 0 (30), +-1/2 (20+20)
    print(f"(1) 600-cell: {n} vertices, {len(edges)} edges at 36 degrees, 12-regular; "
          f"71 root-compatible partners per vertex  [{time.time()-t0:.2f}s]")

    # (2) the inscribed 24-cells
    cl0 = enumerate_cliques(cadj, n, 24, fixed=0)
    cells = set()
    # each clique through vertex 0 is a 24-cell; collect all 25 by moving the fixed vertex
    for f in range(n):
        for c in enumerate_cliques(cadj, n, 24, fixed=f):
            cells.add(c)
        if len(cells) == 25 and f >= 4: break
    cells = sorted(cells)
    assert len(cl0) == 5, len(cl0)
    assert len(cells) == 25, len(cells)
    for c in cells:
        # independent in the edge graph, and pairwise root-like
        mem = [i for i in range(n) if c >> i & 1]
        for i, j in itertools.combinations(mem, 2):
            assert not (adj[i] >> j & 1)
            assert inner4(V[i], V[j]) in rootlike
    print(f"(2) inscribed 24-cells: {len(cells)} (5 through each vertex), each an independent "
          f"set with inner products in {{-1,-1/2,0,1/2}}  [{time.time()-t0:.2f}s]")

    # (3) independence number 24
    ind25 = enumerate_independent(adj, n, 25, fixed=0)
    assert len(ind25) == 0
    ind24 = enumerate_independent(adj, n, 24, fixed=0)
    assert len(ind24) == 5 and set(ind24) == set(cl0)
    print(f"(3) independent sets through vertex 0: none of size 25, exactly 5 of size 24, "
          f"and those are the five 24-cells through it  [{time.time()-t0:.2f}s]")

    # (4) every independent 23-set lies in a 24-cell
    ind23 = enumerate_independent(adj, n, 23, fixed=0)
    outside = [c for c in ind23 if not any((c & ~cell) == 0 for cell in cells)]
    print(f"(4) independent sets of size 23 through vertex 0: {len(ind23)} "
          f"(so {len(ind23)*120//23} in all, against 25 x 24 = 600); "
          f"lying in no 24-cell: {len(outside)}  [{time.time()-t0:.2f}s]")
    assert len(ind23) == 115 and len(ind23) * 120 % 23 == 0 and len(outside) == 0

    with open("cell600_graph.txt", "w") as f:
        f.write(f"{n} {len(edges)}\n")
        for i, j in edges: f.write(f"{i} {j}\n")
        f.write(f"24cells {len(cells)}\n")
        for c in cells:
            f.write(" ".join(str(i) for i in range(n) if c >> i & 1) + "\n")
    print("graph and cell list written to cell600_graph.txt for cell600_enum.c")

    if "--sat" in sys.argv:
        from pysat.solvers import Cadical153
        from pysat.card import CardEnc, EncType
        from pysat.formula import IDPool
        lits = list(range(1, n+1)); pool = IDPool(start_from=n+1)
        cl = [[-(i+1), -(j+1)] for i, j in edges]
        cl += CardEnc.atleast(lits=lits, bound=23, vpool=pool, encoding=EncType.seqcounter).clauses
        for c in cells:
            cl.append([i+1 for i in range(n) if not (c >> i & 1)])
        with Cadical153(bootstrap_with=cl) as s:
            sat = s.solve()
        print(f"SAT cross-check: {'SATISFIABLE' if sat else 'UNSATISFIABLE'}  [{time.time()-t0:.1f}s]")
        assert not sat
    print("PASS: every 23-point code of minimal angle 60 degrees among the vertices of the "
          "600-cell is an inscribed 24-cell with one vertex deleted.")

if __name__ == "__main__":
    main()

"""
c28_config_test.py -- the certificate of prop:count28-few on actual sets of 28 directions and
distances: U(Y) = sum S - sum Pi must be at most
B(Y) = sum_y (f(|y|) + p_type(y)) + t/2 + sum_pairs (K - Pi + PAIR3) + sum_triples TRIPLE3,
since B - U is the sum of the two kernels over Y.  Run on the 12-close witness of the
direction search, on the root system with four centres in deep holes, and on random sets.
Floating point; a check of the implementation, used in no proof.
"""
import sys, os, itertools
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.argv = ['x', '10', '1', REPO + '/gap_closure/CM/case28_lo13.json']
sys.path.insert(0, REPO + '/gap_closure/CM')
import combo_gen2 as G
T3 = G.T3
Z = np.load(REPO + '/gap_closure/CM/c28lo13b_d8_r1.npz')
x3 = np.array(Z['x3'], float); A = np.array(Z['A'], float); z = np.array(Z['z'], float); t = float(Z['t'])
L = T3.Layout(3, 8)
TI = {'A': 0, 'B': 1, 'F': 2}
def typ(d):
    return 'A' if d <= 2.05 else ('B' if d <= 2.35 else 'F')
def K(d1, d2, u):
    return float(np.einsum('k,a,kab,b->', G.ubasis(np.array([u]), G.D2)[0], G.pbasis(np.array([d1]), G.R2)[0], A, G.pbasis(np.array([d2]), G.R2)[0]))
def row(fn, *args):
    B = T3.Builder(L); fn(B, np.arange(1), *args); return float((B.matrix(1) @ x3)[0])
def test(W, d, label):
    W = W / np.linalg.norm(W, axis=1)[:, None]; n = len(d)
    Gm = W @ W.T
    ty = [typ(x) for x in d]
    U = sum(G.S(np.array([x]))[0] for x in d)
    Bv = t / 2
    for i in range(n):
        p = G.pbasis(np.array([d[i]]), G.R2)[0]
        Bv += G.S(np.array([d[i]]))[0] + K(d[i], d[i], 1.0) / 2 - z @ p + row(T3.add_point, TI[ty[i]])
    worst_pair = -1
    for i, j in itertools.combinations(range(n), 2):
        u = min(1.0, Gm[i, j])
        Pi = G.pair(np.array([d[i] / 2]), np.array([d[j] / 2]), np.array([u]))[0]
        U -= Pi
        Bv += K(d[i], d[j], u) - Pi + row(T3.add_pair, TI[ty[i]], TI[ty[j]], np.array([u]))
        worst_pair = max(worst_pair, u - G.amax(d[i], d[j]))
    for i, j, k in itertools.combinations(range(n), 3):
        Bv += row(T3.add_triple, (TI[ty[i]], TI[ty[j]], TI[ty[k]]), np.array([Gm[i, j]]), np.array([Gm[i, k]]), np.array([Gm[j, k]]))
    print('%s: largest pair excess %.2e, U = %.6f, B = %.6f, B - U = %.6f' % (label, worst_pair, U, Bv, Bv - U), flush=True)
wit = np.loadtxt(REPO + '/gap_closure/CM/c28_close12_witness.txt')
test(wit[:, :4], wit[:, 4], 'witness, 12 close')
# the root system with four further centres near sqrt 6 in holes of it (not a packing)
roots = []
for i, j in itertools.combinations(range(4), 2):
    for si in (1, -1):
        for sj in (1, -1):
            v = np.zeros(4); v[i] = si; v[j] = sj; roots.append(v)
roots = np.array(roots)
holes = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
test(np.r_[roots, holes], np.r_[[2.0] * 24, [2.4494] * 4], 'D4 + 4 deep holes (pairs violate the packing condition)')
rng = np.random.default_rng(5)
for s in range(3):
    W = rng.standard_normal((28, 4)); d = 2 + (6 ** .5 - 2) * rng.random(28)
    test(W, d, 'random %d' % s)

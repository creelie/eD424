#!/usr/bin/env python3
"""
est_binrefine.py -- how much a certificate of combo_gen2.py / combo_direct.py gains when
its pair and triple thresholds are taken per combination of bins instead of per
combination of types.

The bound of a certificate is
    sum_b n_b (m_b + p_type(b)) + t/2 + sum_{s<=t} N_st c_st + sum_{s<=t<=r} N_str c_str,
with c_st the largest value of K - Pi + PAIR3_st over the admissible pairs of types
(s, t), and c_str that of TRIPLE3 over the admissible triples.  The same kernel also
gives
    sum_b n_b (m_b + p_type(b)) + t/2 + sum_{b<=b'} N_bb' c_bb' + sum_{b<=b'<=b''} N_bb'b'' c_bb'b'',
with c_bb' the largest value over pairs whose distances lie in the bins b and b' (so
that the inner product is at most a(hi_b, hi_b') rather than a at the ends of the type
ranges), and c_bb'b'' likewise for triples.  The second is never larger than the first.
Both maxima are found here by sampling and local ascent, in floating point: the script
measures the gain and proves nothing.

Usage: python3 est_binrefine.py d3 0 case.json cert.npz
"""
import itertools
import os
import sys
from math import comb

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import combo_gen2 as G  # noqa: E402  (reads the case file from argv[3])
from combo_gen2 import T3, TYPES, BINS, COUNTS, amax, pair_values, triple_values, climb, gram_ok  # noqa: E402

d3 = int(sys.argv[1])
sv = np.load(sys.argv[4])
x3v, Av, t, m = sv['x3'], sv['A'], float(sv['t']), sv['m']
L = T3.Layout(len(TYPES), d3)
ti = {s: i for i, s in enumerate(TYPES)}
rng = np.random.default_rng(7)

B = T3.Builder(L)
for s in TYPES:
    T3.add_point(B, np.array([ti[s]]), ti[s])
pvec = dict(zip(TYPES, B.matrix(len(TYPES)) @ x3v))

nb = len(BINS)
btype = [b[0] for b in BINS]


def pair_max(b1, b2, n=20000):
    (s, lo1, hi1), (tt, lo2, hi2) = BINS[b1], BINS[b2]
    p = lo1 + (hi1 - lo1) * rng.random(n); q = lo2 + (hi2 - lo2) * rng.random(n)
    k = n // 6
    p[:k] = lo1; q[k:2 * k] = lo2; p[2 * k:3 * k] = hi1; q[3 * k:4 * k] = hi2
    top = amax(p, q)
    u = -1 + (top + 1) * rng.random(n) ** 0.5
    u[4 * k:5 * k] = top[4 * k:5 * k] - 3e-3 * rng.random(k)
    X = np.stack([p, q, u], 1)
    v = np.concatenate([pair_values((s, tt), X[i:i + 4000], x3v, Av, L, 0.0) for i in range(0, n, 4000)])
    starts = X[np.argsort(v)[-40:]]

    def proj(Y):
        Y[:, 0] = np.clip(Y[:, 0], lo1, hi1); Y[:, 1] = np.clip(Y[:, 1], lo2, hi2)
        Y[:, 2] = np.clip(Y[:, 2], -1, amax(Y[:, 0], Y[:, 1]))
        return Y
    _, val = climb(lambda Y: pair_values((s, tt), Y, x3v, Av, L, 0.0), starts, proj, lambda Y: np.ones(len(Y), bool))
    return max(v.max(), val.max())


def triple_max(bs, n=6000):
    cb = tuple(btype[b] for b in bs)
    hi = [BINS[b][2] for b in bs]
    T = np.array([amax(hi[0], hi[1]), amax(hi[0], hi[2]), amax(hi[1], hi[2])])
    g = np.r_[T3.random_triples(T[0], T[1], T[2], n), T3.triple_grid(T[0], T[1], T[2], 14, 14, 6)]
    v = np.concatenate([triple_values(cb, g[i:i + 4000], x3v, L, 0.0) for i in range(0, len(g), 4000)])
    starts = g[np.argsort(v)[-30:]]
    _, val = climb(lambda Y: triple_values(cb, Y, x3v, L, 0.0), starts, lambda Y: np.clip(Y, -1, T), gram_ok)
    return max(v.max(), val.max())


c2b = {}
for b1, b2 in itertools.combinations_with_replacement(range(nb), 2):
    c2b[(b1, b2)] = pair_max(b1, b2)
print('pairs done', flush=True)
c3b = {}
for bs in itertools.combinations_with_replacement(range(nb), 3):
    c3b[bs] = triple_max(bs)
print('triples done', flush=True)

# type-level thresholds from the same maxima, so that the comparison is like for like
c2t, c3t = {}, {}
for (b1, b2), v in c2b.items():
    k = (btype[b1], btype[b2]); c2t[k] = max(c2t.get(k, -np.inf), v)
for bs, v in c3b.items():
    k = tuple(btype[b] for b in bs); c3t[k] = max(c3t.get(k, -np.inf), v)


def Nmulti(n, bs):
    out = 1
    for b in set(bs):
        out *= comb(n[b], bs.count(b))
    return out


best_t = best_b = -np.inf
arg_t = arg_b = None
for n in COUNTS:
    base = sum(n[b] * (m[b] + pvec[btype[b]]) for b in range(nb)) + t / 2
    vb = base + sum(Nmulti(n, list(k)) * v for k, v in c2b.items()) + sum(Nmulti(n, list(k)) * v for k, v in c3b.items())
    tc = {s: sum(n[b] for b in range(nb) if btype[b] == s) for s in TYPES}
    vt = base + sum(G.Npair(tc, *k) * v for k, v in c2t.items()) + sum(G.Ntriple(tc, list(k)) * v for k, v in c3t.items())
    if vt > best_t:
        best_t, arg_t = vt, n
    if vb > best_b:
        best_b, arg_b = vb, n
print('file bound %.5f' % float(sv['bound']))
print('type thresholds (sampled maxima): %.5f at %s' % (best_t, arg_t))
print('bin thresholds  (sampled maxima): %.5f at %s' % (best_b, arg_b))
print('pair thresholds by bins:', {k: round(v, 6) for k, v in c2b.items()})

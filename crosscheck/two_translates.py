#!/usr/bin/env python3
"""
two_translates.py -- the statements (G) and (C) of d4-voronoi-cells on packings
whose centres form two translates of a lattice, computed with the code of both
packages.

A two-periodic set Lambda u (Lambda + b) in R^4 is given, as in the paper in
../paper, by J = [[Q, r], [r^T, s]] with Q the Gram matrix of a basis a_1..a_4
of Lambda, r_i = <a_i, b> and s = |b|^2; it lies in R^4 when det J = 0, and its
minimum distance is 2 when min J[k] = 4 over k in M* (verification/data).  The
map x -> b - x exchanges the two translates, so every centre sees the same
configuration and the centre 0 stands for all.  For each packing the script
computes

  * the cell volume twice: covol(Lambda)/2 = sqrt(det Q)/2 from the Gram
    matrix, and the volume of the Voronoi polytope of 0 from its half-spaces
    (scipy), which must agree;
  * Y, the centres within sqrt 6 of 0, and T(Y) with T from
    d4-voronoi-cells/multi_cap/truncated_search.py;
  * when |Y| = 24 and T(Y) <= 8, the volume of V(Y) n K(Y), with K(Y) the
    convex hull of 0 and the points 4y/|y|^2: statement (G) asks for >= 8;
  * when |Y| >= 25, whether T(Y) > 8: statement (C);
  * whether vol(V_0) >= vol(V(Y) n K(Y)), which prop:inversion-hull proves.

The packings are the vertices and edge points of R(4) with det J = 0 (the
candidates of Proposition 5.4 of the paper, found here in floating point),
random two-periodic packings, and random perturbations of sqrt2 D4 of three
sizes.  None of these has 25 centres within sqrt 6, so a last part looks for
such packings on purpose: starting near sqrt2 D4 = 2Z^4 u (2Z^4 + (1,1,1,1)),
it keeps the 24 minimal vectors and one vector of norm 8 within sqrt 6 and
the minimum distance at 2, and minimises T by sequential quadratic
programming, once for each of the 24 vectors of norm 8.  Floating point throughout: a consistency check of the two packages
and a test of (G) and (C) on a family where the paper's theorem gives the
conclusion, not a proof of anything.

Usage: python3 crosscheck/two_translates.py [random] [near] [seed]
"""
import itertools
import json
import math
import os
import sys
from fractions import Fraction

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'd4-voronoi-cells', 'multi_cap'))
import truncated_search as ts  # noqa: E402

LEVEL = 8.0
R6 = math.sqrt(6.0)


# ---------------------------------------------------------------- the packing
def lll(B, delta=0.75):
    """LLL-reduce the columns of B (floating point, four columns)."""
    B = B.copy()
    n = B.shape[1]

    def gso(B):
        Bs = B.copy()
        mu = np.zeros((n, n))
        for i in range(n):
            for j in range(i):
                mu[i, j] = B[:, i] @ Bs[:, j] / (Bs[:, j] @ Bs[:, j])
                Bs[:, i] -= mu[i, j] * Bs[:, j]
        return Bs, mu

    Bs, mu = gso(B)
    k = 1
    while k < n:
        for j in range(k - 1, -1, -1):
            q = round(mu[k, j])
            if q:
                B[:, k] -= q * B[:, j]
                Bs, mu = gso(B)
        if Bs[:, k] @ Bs[:, k] >= (delta - mu[k, k - 1] ** 2) * (Bs[:, k - 1] @ Bs[:, k - 1]):
            k += 1
        else:
            B[:, [k - 1, k]] = B[:, [k, k - 1]]
            Bs, mu = gso(B)
            k = max(k - 1, 1)
    return B


def embed(J):
    """a reduced basis matrix B (columns in Lambda) and b, with Gram matrix Q
    and <a_i, b> = r_i up to the change of basis; b is moved by a vector of
    Lambda close to it, which leaves Lambda + b unchanged."""
    Q, r = J[:4, :4], J[:4, 4]
    B0 = np.linalg.cholesky(Q).T
    b = np.linalg.solve(B0.T, r)
    B = lll(B0)
    b = b - B @ np.round(np.linalg.solve(B, b))
    return B, b


def points_within(B, b, rad, cap=4_000_000):
    """all points of Lambda u (Lambda + b) other than 0 with norm < rad."""
    Binv = np.linalg.inv(B)
    out = []
    for l in (0, 1):
        shift = l * b
        # |B n + shift| < rad  =>  |n_i + (B^-1 shift)_i| <= |row_i(B^-1)| rad
        c = Binv @ shift
        w = np.linalg.norm(Binv, axis=1) * rad
        lo, hi = np.floor(-c - w), np.ceil(-c + w)
        if np.prod(hi - lo + 1) > cap:
            raise ValueError('enumeration box too large')
        rng = [np.arange(lo[i], hi[i] + 1) for i in range(4)]
        grid = np.stack(np.meshgrid(*rng, indexing='ij'), -1).reshape(-1, 4).astype(float)
        P = grid @ B.T + shift
        n2 = np.sum(P * P, axis=1)
        keep = (n2 < rad * rad) & (n2 > 1e-12)
        out.append(P[keep])
    return np.vstack(out)


def normalise(J):
    """scale J so that the minimum distance is 2; return J and the minimum."""
    B, b = embed(J)
    P = points_within(B, b, 1.01 * min(np.min(np.linalg.norm(B, axis=0)), np.linalg.norm(b)))
    m2 = np.min(np.sum(P * P, axis=1))
    # differences between two centres of Lambda + b are vectors of Lambda, so
    # the norms from 0 give the minimum distance
    return J * (4.0 / m2), m2


# ------------------------------------------------------------------- volumes
def poly_volume(hs):
    """volume of {x : A x + c <= 0} for hs = [A | c], with 0 inside."""
    hi = HalfspaceIntersection(hs, np.zeros(4))
    return ConvexHull(hi.intersections).volume


def cell_halfspaces(Y):
    return np.hstack([Y, -0.5 * np.sum(Y * Y, axis=1)[:, None]])


def hull_halfspaces(Y):
    inv = 4.0 * Y / np.sum(Y * Y, axis=1)[:, None]
    return ConvexHull(np.vstack([np.zeros(4), inv])).equations


def voronoi_volume(B, b):
    """volume of the Voronoi cell of 0, from the centres within twice its
    circumradius (enlarged until that holds)."""
    rad = 4.0
    while True:
        P = points_within(B, b, rad)
        try:
            hi = HalfspaceIntersection(cell_halfspaces(P), np.zeros(4))
        except QhullError:               # the centres found so far leave the cell unbounded
            rad *= 1.5
            continue
        circ = np.max(np.linalg.norm(hi.intersections, axis=1))
        if not np.isfinite(circ):
            rad *= 1.5
            continue
        if 2 * circ < rad:
            return ConvexHull(hi.intersections).volume
        rad = 2 * circ + 0.5


# ------------------------------------------------------------------ packings
def fdet(M):
    """determinant of a square matrix of Fractions."""
    M = [row[:] for row in M]
    n, d = len(M), Fraction(1)
    for i in range(n):
        p = next((k for k in range(i, n) if M[k][i] != 0), None)
        if p is None:
            return Fraction(0)
        if p != i:
            M[i], M[p] = M[p], M[i]
            d = -d
        d *= M[i][i]
        for k in range(i + 1, n):
            f = M[k][i] / M[i][i]
            M[k] = [a - f * c for a, c in zip(M[k], M[i])]
    return d


def det_poly(J, X):
    """exact coefficients (highest first) of t -> det(J + t X), degree <= 5."""
    pts = [Fraction(k) for k in range(6)]
    vals = [fdet([[a + t * c for a, c in zip(ra, rc)] for ra, rc in zip(J, X)]) for t in pts]
    coef = [Fraction(0)] * 6                 # Lagrange interpolation, lowest first
    for i, ti in enumerate(pts):
        basis = [Fraction(1)]
        den = Fraction(1)
        for j, tj in enumerate(pts):
            if j != i:
                basis = [Fraction(0)] + basis
                for k in range(len(basis) - 1):
                    basis[k] -= tj * basis[k + 1]
                den *= ti - tj
        for k in range(6):
            coef[k] += vals[i] * basis[k] / den
    return coef[::-1]


def candidates():
    """vertices and edge points of R(4) with det J = 0: the polynomial
    det(J + tX) is exact, its roots are taken in floating point."""
    data = json.load(open(os.path.join(ROOT, 'verification', 'data', 'r4_vertices_edges.json')))
    frac = lambda M: [[Fraction(x) for x in row] for row in M]  # noqa: E731
    flt = lambda M: np.array([[float(x) for x in row] for row in M])  # noqa: E731
    V = [frac(v['J']) for v in data['vertices']]
    out = []
    for i, J in enumerate(V):
        if fdet(J) == 0:
            out.append(('vertex J_%d' % i, flt(J)))
    for k, e in enumerate(data['edges']):
        J, X = V[e['from']], frac(e['X'])
        hi = Fraction(e['t']) if e['t'] is not None else None
        p = det_poly(J, X)
        while p and p[0] == 0:
            p = p[1:]
        if not p:                            # an edge of packings: take its midpoint
            mid = hi / 2 if hi is not None else Fraction(1)
            out.append(('edge %d of packings from J_%d' % (k, e['from']),
                        flt(J) + float(mid) * flt(X)))
            continue
        if len(p) == 1:
            continue
        for t in np.roots([float(c) for c in p]):
            if abs(t.imag) > 1e-9 * max(1, abs(t)):
                continue
            t = t.real
            if t <= 1e-9 or (hi is not None and t >= float(hi) - 1e-9):
                continue
            Jt = flt(J) + t * flt(X)
            if np.min(np.linalg.eigvalsh(Jt[:4, :4])) <= 0:
                continue
            out.append(('edge %d from J_%d, t = %.6f' % (k, e['from'], t), Jt))
    return out


def on_surface(Q, r):
    """J with s chosen so that det J = 0."""
    J = np.zeros((5, 5))
    J[:4, :4] = Q
    J[:4, 4] = J[4, :4] = r
    J[4, 4] = r @ np.linalg.solve(Q, r)
    return J


def random_packing(rng):
    E = rng.standard_normal((4, 4))
    Q = np.eye(4) + 0.25 * (E + E.T)
    while np.min(np.linalg.eigvalsh(Q)) < 0.2:
        E = rng.standard_normal((4, 4))
        Q = np.eye(4) + 0.25 * (E + E.T)
    r = Q @ rng.random(4)               # b = sum t_i a_i with t in [0, 1)^4
    return on_surface(Q, r)


def near_d4(rng, eps):
    E = rng.standard_normal((5, 5))
    E = eps * (E + E.T) / 2
    J = J0 + E
    return on_surface(J[:4, :4], J[:4, 4])


J0 = np.array([[4, 0, 0, 0, 2], [0, 4, 0, 0, -2], [0, 0, 4, 0, -2], [0, 0, 0, 4, -2], [2, -2, -2, -2, 4]], float)
IU = np.triu_indices(5)


def forced(rng, starts=2):
    """least T over two-periodic packings near sqrt2 D4 in which the 24
    minimal vectors of D4 and one vector of norm 8 lie within sqrt 6."""
    box = np.array(list(itertools.product(range(-2, 3), repeat=4)), float)
    K = np.vstack([np.hstack([box, np.full((len(box), 1), l)]) for l in (0, 1)])
    K = K[np.any(K != 0, axis=1)]
    n0 = np.einsum('ki,ij,kj->k', K, J0, K)
    mins, second = K[np.abs(n0 - 4) < 1e-9], K[np.abs(n0 - 8) < 1e-9]

    def J_of(x):
        J = np.zeros((5, 5))
        J[IU] = x
        J = J + J.T - np.diag(np.diag(J))
        return on_surface(J[:4, :4], J[:4, 4])

    def norms(x, KK):
        return np.einsum('ki,ij,kj->k', KK, J_of(x), KK)

    def centres(x):
        return points_within(*embed(J_of(x)), R6)

    out = []
    for extra in second:
        tgt = np.vstack([mins, extra[None]])
        cons = [{'type': 'ineq', 'fun': lambda x: norms(x, K) - 4},
                {'type': 'ineq', 'fun': lambda x, tgt=tgt: 6 - 1e-6 - norms(x, tgt)}]
        for _ in range(starts):
            E = rng.standard_normal((5, 5))
            x0 = (J0 + 0.025 * (E + E.T))[IU]
            r = minimize(lambda x: float(ts.T(centres(x))), x0, method='SLSQP', constraints=cons,
                         options={'maxiter': 400, 'ftol': 1e-12})
            if min(np.min(norms(r.x, K) - 4), np.min(6 - norms(r.x, tgt))) < -1e-8:
                continue
            Y = centres(r.x)
            out.append((float(ts.T(Y)), len(Y), float(np.linalg.det(J_of(r.x)[:4, :4])),
                        np.sort(np.linalg.norm(Y, axis=1))))
    return out


# ------------------------------------------------------------------ the tests
def examine(name, J):
    J, _ = normalise(J)
    B, b = embed(J)
    covol_half = math.sqrt(np.linalg.det(J[:4, :4])) / 2
    vol = voronoi_volume(B, b)
    Y = points_within(B, b, R6)
    T = ts.T(Y)
    row = {'name': name, 'detQ': float(np.linalg.det(J[:4, :4])), 'cell_gram': covol_half,
           'cell_poly': vol, 'n': len(Y), 'T': float(T), 'VK': None}
    try:
        row['VK'] = poly_volume(np.vstack([cell_halfspaces(Y), hull_halfspaces(Y)]))
    except QhullError:                   # too few centres within sqrt 6 for K(Y) to be a body
        pass
    return row


def main():
    nrand = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    nnear = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 1)

    packs = candidates()
    print('%d candidate packings from the vertex and edge table' % len(packs))
    packs += [('random %d' % k, random_packing(rng)) for k in range(nrand)]
    for eps in (0.01, 0.05, 0.2):
        packs += [('near D4, eps %.2f, %d' % (eps, k), near_d4(rng, eps)) for k in range(nnear)]
    rows, skipped = [], 0
    for n, J in packs:
        try:
            rows.append(examine(n, J))
        except ValueError:
            skipped += 1
            print('  skipped: %s' % n)
    if skipped:
        print('%d packings skipped: their cells are too long for the enumeration' % skipped)

    worst_cell = max(abs(r['cell_gram'] - r['cell_poly']) / r['cell_gram'] for r in rows)
    print('cell volume from the Gram matrix and from the half-spaces: largest relative difference %.1e' % worst_cell)
    print('least cell volume: %.6f (the paper proves >= 8, equality only for sqrt2 D4)'
          % min(r['cell_gram'] for r in rows))
    hull = [r for r in rows if r['VK'] is not None]
    bad = [r for r in hull if r['VK'] > r['cell_poly'] * (1 + 1e-9)]
    print('prop:inversion-hull, vol(V_0) >= vol(V(Y) n K(Y)): %d packings, %d failures' % (len(hull), len(bad)))

    counts = {}
    for r in rows:
        counts[r['n']] = counts.get(r['n'], 0) + 1
    print('number of centres within sqrt 6: ' + ', '.join('%d: %d packings' % kv for kv in sorted(counts.items())))

    G = [r for r in rows if r['n'] == 24 and r['T'] <= LEVEL]
    print('(G): %d packings with |Y| = 24 and T <= 8' % len(G))
    if G:
        g = min(G, key=lambda r: r['VK'])
        print('     least vol(V(Y) n K(Y)) = %.6f at %s (T = %.6f, det Q = %.4f)' % (g['VK'], g['name'], g['T'], g['detQ']))
        print('     (G) fails at %d of them' % sum(r['VK'] < LEVEL - 1e-9 for r in G))
    C = [r for r in rows if r['n'] >= 25]
    print('(C): %d packings with |Y| >= 25' % len(C))
    if C:
        c = min(C, key=lambda r: r['T'])
        print('     least T = %.6f at %s (|Y| = %d)' % (c['T'], c['name'], c['n']))
        print('     (C) fails at %d of them' % sum(r['T'] <= LEVEL for r in C))
    F = forced(rng)
    print('25 centres within sqrt 6 on purpose: %d local minima of T from %d starts' % (len(F), 48))
    if F:
        f = min(F)
        print('     least T = %.6f, with |Y| = %d and det Q = %.4f; distances:' % (f[0], f[1], f[2]))
        print('     ' + ' '.join('%.4f' % d for d in f[3]))
        print('     (C) fails at %d of them' % sum(t <= LEVEL for t, *_ in F))
    lowT = min(rows, key=lambda r: r['T'])
    print('least T over all packings: %.6f at %s (|Y| = %d, cell %.6f)' % (lowT['T'], lowT['name'], lowT['n'], lowT['cell_gram']))
    print('floating point: a consistency check, not a proof')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
shell_neighbour_search.py -- exploration, not proof: look for a unit-ball
packing in R^4 whose central Voronoi cell has volume below 8 in the one
regime the proof of the paper leaves open, a centre with non-contact
neighbours in the shell (2, 2 sqrt 2) whose distance profile the covering
criterion of shell_reduction.py does not settle (at least 23 neighbours
within 2 sqrt 2, some of them near-contacts).

A configuration is n neighbour centres y_1..y_n in R^4 around the centre 0,
with |y_i| >= 2 and |y_i - y_j| >= 2.  The objective is the exact volume of
the Voronoi cell {x : <x, y_i> <= |y_i|^2 / 2}, computed as a polytope, with
its exact gradient: moving y_i moves facet i with normal velocity
(d(|y|^2/2) - <x, dy>)/|y|, so

    d vol / d y_i = (area_i / |y_i|) (y_i - centroid_i),

area_i and centroid_i being the 3-volume and the centroid of facet i.  The
packing constraints are imposed exactly (SLSQP with analytic Jacobians), and
every endpoint is re-checked for feasibility to 1e-9 before it is reported.

Starts, in rotation:
  near24   the 24 roots of D4 at distance 2, each pushed out to 2 + U(0, s)
           and turned by a random angle up to a, with extra centres added in
           the largest holes;
  del23    the 23 roots of a deletion plus 1 to 3 centres near the deleted
           root at distances in (2, 2.4);
  random   n centres at random directions and distances in [2, 2.4],
           made feasible by the first solve.

    python3 shell_neighbour_search.py [starts] [seed] [processes] [delta]

With a fifth argument delta > 0 one neighbour is held at distance exactly
2 + delta throughout (an equality constraint), so that the search measures the
least volume of a cell that has a centre in the shell at that distance; this
is the regime of near-contacts that the covering criterion leaves open.

Reports, for every start, the endpoint's volume, its number of contacts and
of shell neighbours, and whether the covering criterion Phi already settles
its distance profile; then the least volume over feasible endpoints with at
least one shell neighbour.  The expected answer, if the conjecture is true,
is that nothing goes below 8.
"""
import itertools
import math
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, HalfspaceIntersection

S2 = 2 * math.sqrt(2)
TOL = 1e-9


def roots():
    out = []
    for i, j in itertools.combinations(range(4), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(4)
                v[i], v[j] = si, sj
                out.append(v / math.sqrt(2))
    return np.array(out)


R24 = roots()


def cell(Y):
    """vertices, hull and halfspaces of the cell {x : <x, y> <= |y|^2/2}"""
    b = 0.5 * (Y * Y).sum(1)
    hs = np.hstack([Y, -b[:, None]])
    P = HalfspaceIntersection(hs, np.zeros(4)).intersections
    return P, ConvexHull(P), b


def volume_and_grad(Y):
    try:
        P, H, b = cell(Y)
    except Exception:
        return None, None
    vol = H.volume
    if not np.isfinite(vol) or vol > 1e4:
        return None, None
    grad = np.zeros_like(Y)
    norms = np.linalg.norm(Y, axis=1)
    U = Y / norms[:, None]
    for simplex, eq in zip(H.simplices, H.equations):
        n = eq[:4]
        # which half-space does this facet lie on?  the one whose unit normal matches
        k = int(np.argmax(U @ n))
        if U[k] @ n < 1 - 1e-7:
            continue
        V = P[simplex]
        E = V[1:] - V[0]
        g = E @ E.T
        a = math.sqrt(max(np.linalg.det(g), 0.0)) / 6.0
        c = V.mean(0)
        grad[k] += (a / norms[k]) * (Y[k] - c)
    return vol, grad


def objective(z, n):
    Y = z.reshape(n, 4)
    v, g = volume_and_grad(Y)
    if v is None:
        return 1e3, np.zeros_like(z)
    return v, g.ravel()


def constraints(n):
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def f(z):
        Y = z.reshape(n, 4)
        c1 = (Y * Y).sum(1) - 4.0
        D = Y[[i for i, _ in pairs]] - Y[[j for _, j in pairs]]
        c2 = (D * D).sum(1) - 4.0
        return np.concatenate([c1, c2])

    def J(z):
        Y = z.reshape(n, 4)
        rows = np.zeros((n + len(pairs), 4 * n))
        for i in range(n):
            rows[i, 4 * i:4 * i + 4] = 2 * Y[i]
        for k, (i, j) in enumerate(pairs):
            d = 2 * (Y[i] - Y[j])
            rows[n + k, 4 * i:4 * i + 4] = d
            rows[n + k, 4 * j:4 * j + 4] = -d
        return rows

    return {'type': 'ineq', 'fun': f, 'jac': J}


def random_rotation(rng, angle):
    A = rng.normal(size=(4, 4))
    A = A - A.T
    A *= angle / max(np.linalg.norm(A, 2), 1e-12)
    w, V = np.linalg.eig(A)
    return np.real(V @ np.diag(np.exp(w)) @ np.linalg.inv(V))


def holes():
    """the 24 deep-hole directions of the root system (vertices of the dual 24-cell)"""
    out = [np.eye(4)[i] * s for i in range(4) for s in (1, -1)]
    out += [np.array(v) / 2 for v in itertools.product((1, -1), repeat=4)]
    return np.array(out)


def start(kind, rng):
    if kind == 'near24':
        s = rng.uniform(0.0, 0.15)
        a = rng.uniform(0.0, 0.15)
        Y = []
        for r in R24:
            u = r + rng.normal(size=4) * a
            u /= np.linalg.norm(u)
            Y.append(u * (2 + rng.uniform(0, s)))
        extra = rng.integers(0, 3)
        Hh = holes()
        for h in Hh[rng.choice(len(Hh), extra, replace=False)]:
            Y.append(h * rng.uniform(2.6, 2.9))
        return np.array(Y)
    if kind == 'del23':
        k = rng.integers(0, 24)
        W = np.delete(R24, k, axis=0) * 2
        r0 = R24[k]
        extra = []
        for _ in range(rng.integers(1, 4)):
            u = r0 + rng.normal(size=4) * 0.25
            u /= np.linalg.norm(u)
            extra.append(u * rng.uniform(2.0, 2.4))
        Y = np.vstack([W, np.array(extra)])
        return Y @ random_rotation(rng, rng.uniform(0, 0.1)).T
    n = rng.integers(22, 27)
    U = rng.normal(size=(n, 4))
    U /= np.linalg.norm(U, axis=1)[:, None]
    return U * rng.uniform(2.0, 2.4, size=(n, 1))


def run(args):
    idx, seed, delta = args
    rng = np.random.default_rng(seed)
    kind = ('near24', 'del23', 'random')[idx % 3]
    Y0 = start(kind, rng)
    n = len(Y0)
    cons_list = [constraints(n)]
    if delta > 0:
        # hold neighbour 0 at distance exactly 2 + delta
        Y0[0] *= (2 + delta) / np.linalg.norm(Y0[0])

        def feq(z):
            return np.array([z[0:4] @ z[0:4] - (2 + delta) ** 2])

        def jeq(z):
            row = np.zeros((1, 4 * n))
            row[0, 0:4] = 2 * z[0:4]
            return row

        cons_list.append({'type': 'eq', 'fun': feq, 'jac': jeq})
    t0 = time.time()
    res = minimize(objective, Y0.ravel(), args=(n,), jac=True, method='SLSQP',
                   constraints=cons_list,
                   options={'maxiter': 400, 'ftol': 1e-12})
    Y = res.x.reshape(n, 4)
    cons = constraints(n)['fun'](res.x)
    feasible = cons.min() >= -TOL
    if delta > 0:
        feasible = feasible and abs(np.linalg.norm(Y[0]) - (2 + delta)) < 1e-7
    v, _ = volume_and_grad(Y)
    d = np.sort(np.linalg.norm(Y, axis=1))
    contacts = int((d < 2 + 1e-6).sum())
    shell = int(((d > 2 + 1e-6) & (d < S2)).sum())
    return dict(idx=idx, kind=kind, n=n, vol=v, feasible=bool(feasible),
                viol=float(-min(cons.min(), 0.0)), contacts=contacts, shell=shell,
                dists=d[d < S2], time=time.time() - t0)


def main():
    starts = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    delta = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    ss = np.random.SeedSequence(seed).spawn(starts)
    jobs = [(i, int(s.generate_state(1)[0]), delta) for i, s in enumerate(ss)]
    print('starts %d (seed %d), kinds near24 / del23 / random in rotation%s'
          % (starts, seed, ('; one neighbour held at distance 2 + %g' % delta) if delta > 0 else ''))
    print('%4s %-7s %3s %12s %9s %9s %6s %s' % ('#', 'kind', 'n', 'volume', 'contacts', 'shell',
                                             'feas', 'shell distances'))
    out = []
    t0 = time.time()
    with Pool(procs) as pool:
        for r in pool.imap_unordered(run, jobs):
            out.append(r)
            sd = ' '.join('%.4f' % x for x in r['dists'] if x > 2 + 1e-6)
            print('%4d %-7s %3d %12s %9d %9d %6s %s' % (
                r['idx'], r['kind'], r['n'],
                ('%.9f' % r['vol']) if r['vol'] is not None else 'unbounded',
                r['contacts'], r['shell'], 'yes' if r['feasible'] else 'no', sd[:60]),
                flush=True)
    good = [r for r in out if r['feasible'] and r['vol'] is not None]
    withshell = [r for r in good if r['shell'] > 0]
    print()
    print('feasible endpoints: %d of %d' % (len(good), len(out)))
    if good:
        b = min(good, key=lambda r: r['vol'])
        print('least volume overall: %.9f (%d contacts, %d shell neighbours)'
              % (b['vol'], b['contacts'], b['shell']))
    if withshell:
        b = min(withshell, key=lambda r: r['vol'])
        print('least volume with a shell neighbour: %.9f (%d contacts, %d shell neighbours at %s)'
              % (b['vol'], b['contacts'], b['shell'],
                 ', '.join('%.4f' % x for x in b['dists'] if x > 2 + 1e-6)))
    below = [r for r in good if r['vol'] < 8 - 1e-7]
    print('RESULT: %s' % ('%d feasible endpoints below 8' % len(below) if below
                          else 'no feasible endpoint has volume below 8 (exploration, not proof)'))
    print('total time %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()

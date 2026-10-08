"""
check_psd.py -- are the zonal matrices positive semidefinite kernels?

The whole point of Z_lambda is that

    ((J1, t1), (J2, t2))  |-->  Z_lambda(J1, J2)_{t1, t2}

is a positive semidefinite kernel on the subsets J of size at most two of any
set of points on the sphere: that is what makes the semidefinite program a
relaxation at all.  Nothing in the way the matrices are computed here forces
this, so it is a real test, and in particular it tests the scalars of Section
3.1, which the comparison with the Gegenbauer polynomials only reaches for the
signatures with lambda2 = 0.

Points are drawn at random on S^3 and the matrix is formed in floating point.
"""
import sys
from fractions import Fraction as Q

import numpy as np

import zonal
from verify45 import irreps, weven, admissible_tuples, D2


def build(ps, lam, X):
    """the big matrix for the point set X (rows of unit vectors)"""
    N = len(X)
    G = X @ X.T
    ws = weven(lam)
    dt = (D2 - sum(lam)) // 2
    subs = [()] + [(i,) for i in range(N)] + \
           [(i, j) for i in range(N) for j in range(i + 1, N)]
    order, index = [], {}
    for i in (0, 1, 2):
        for jk in admissible_tuples(lam, i, ws, dt):
            index[(i, jk[0], jk[1])] = len(order)
            order.append((i, jk[0], jk[1]))
    rows = []
    for J in subs:
        for t in order:
            if t[0] == len(J):
                rows.append((J, t))
    n = len(rows)
    if n == 0:
        return None
    M = np.zeros((n, n))
    cache = {}
    for a, (J1, t1) in enumerate(rows):
        for b, (J2, t2) in enumerate(rows):
            if b < a:
                continue
            i1, j1, k1 = t1
            i2, j2, k2 = t2

            def f(Ja, Jb, p, q):
                return G[Ja[min(p, len(Ja) - 1)], Jb[min(q, len(Jb) - 1)]]

            if i1 > 0 and i2 > 0:
                ips = [f(J1, J1, 0, 1), f(J2, J2, 0, 1), f(J1, J2, 0, 0),
                       f(J1, J2, 0, 1), f(J1, J2, 1, 0), f(J1, J2, 1, 1)]
            else:
                ips = [1.0] * 6
                if i1 > 0:
                    ips[0] = f(J1, J1, 0, 1)
                elif i2 > 0:
                    ips[1] = f(J2, J2, 0, 1)
            key = (tuple(ips), k1, k2)
            if key not in cache:
                cache[key] = zonal.evaluate_zonal_matrix(
                    ps, lam, k1, k2, ips, 1.0, 0.0, float)
            v = ips[0] ** j1 * ips[1] ** j2 * cache[key]
            M[a, b] = M[b, a] = v
    return M


def main(psfile, npoints=4, trials=3, seed=11):
    ps = zonal.load_ps(psfile)
    have = []
    for lam in irreps():
        ws = weven(lam)
        if ws and all((lam, max(a, b), min(a, b)) in ps for a in ws for b in ws):
            have.append(lam)
    rng = np.random.default_rng(seed)
    worst = {}
    for lam in have:
        w = 0.0
        for _ in range(trials):
            X = rng.normal(size=(npoints, 4))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            M = build(ps, lam, X)
            if M is None:
                continue
            ev = np.linalg.eigvalsh(M)
            scale = max(1.0, abs(ev).max())
            w = min(w, ev.min() / scale)
        worst[lam] = w
    bad = [(lam, w) for lam, w in worst.items() if w < -1e-8]
    for lam in sorted(worst):
        print('lambda %-9s  least eigenvalue / scale = %11.3e %s'
              % (str(list(lam)), worst[lam], 'FAIL' if worst[lam] < -1e-8 else ''))
    print()
    print('%d signatures tested, %d with a negative eigenvalue'
          % (len(worst), len(bad)))
    return not bad


if __name__ == '__main__':
    ok = main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    sys.exit(0 if ok else 1)

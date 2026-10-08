#!/usr/bin/env python3
"""
tilt_quartic.py -- is the root system a local minimum of the contact-cell volume
vol{x : <x, w_i> <= 1} among all sets of 24 directions?  (floating point,
exploration; qhull volumes.)

By tilt_block.py the second variation under pure tilts is positive
semidefinite, zero exactly on the six rotations and on nine further
directions F.  Near the root system write the tilt as c = t k + y, with k a
unit vector of F and y in the orthogonal complement of the kernel; the
question is whether (vol - 8)/t^4 stays bounded below by a positive constant
when y is chosen to make the volume as small as possible (the transverse
directions can lower the quartic term through the cubic one).  For
t = 0.02, 0.05, 0.1 the script minimises (vol - 8)/t^4 jointly over k (on the
unit sphere of F) and y (57 dimensions), from 12 random starts each
(L-BFGS with central differences), and prints the least value found, the
unrelaxed value along the same k, and |y|/t^2.
Usage: python3 tilt_quartic.py [starts]
"""
import itertools
import sys
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import HalfspaceIntersection, ConvexHull
import rational_model as RM

A = np.array([[float(x) for x in r] for r in RM.ROOTS]); U = A / np.sqrt(2)
B = np.array([[[float(x) for x in b] for b in RM.BAS[i]] for i in range(24)])
W = np.array([RM.NORM2[i][k] for i in range(24) for k in range(3)], float)


def dirs(c):
    tau = np.einsum('ik,ikp->ip', c.reshape(24, 3), B) / np.sqrt(2)
    w = U + tau
    return w / np.linalg.norm(w, axis=1, keepdims=True)


def vol(c):
    hs = np.hstack([dirs(c), -np.ones((24, 1))])
    return ConvexHull(HalfspaceIntersection(hs, np.zeros(4)).intersections).volume


def ip(x, y):
    return float(np.sum(W * x * y)) / 2          # the inner product of the tangent fields tau


K = np.load('tilt_kernel.npy')
rots = []
for p, q in itertools.combinations(range(4), 2):
    M = np.zeros((4, 4)); M[p, q] = 1; M[q, p] = -1
    rots.append(np.array([np.dot(M @ A[i], B[i, k]) / RM.NORM2[i][k] for i in range(24) for k in range(3)]))
basis = []
for v in rots + list(K) + list(np.eye(72)):
    v = v.copy()
    for b in basis:
        v -= ip(v, b) * b
    n = np.sqrt(ip(v, v))
    if n > 1e-8:
        basis.append(v / n)
assert len(basis) == 72
FLAT = np.array(basis[6:15]); TRANS = np.array(basis[15:])        # 9 and 57


def objective(z, t):
    v, y = z[:9], z[9:]
    k = v / np.linalg.norm(v)
    c = t * (k @ FLAT) + (t ** 2) * (y @ TRANS)                     # y scaled by t^2
    return (vol(c) - 8) / t ** 4


def grad(z, t, h=1e-5):
    g = np.zeros_like(z)
    for j in range(len(z)):
        e = np.zeros_like(z); e[j] = h
        g[j] = (objective(z + e, t) - objective(z - e, t)) / (2 * h)
    return g


def main():
    starts = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    rng = np.random.default_rng(2026)
    print('t      least (vol-8)/t^4 relaxed   unrelaxed along that k   |y|/t^2   [time]')
    for t in (0.1, 0.05, 0.02):
        best = None; t0 = time.time()
        for s in range(starts):
            z0 = np.concatenate([rng.normal(size=9), np.zeros(57)])
            r = minimize(objective, z0, args=(t,), jac=grad, method='L-BFGS-B',
                         options={'maxiter': 200, 'gtol': 1e-7})
            if best is None or r.fun < best[0]:
                v = r.x[:9] / np.linalg.norm(r.x[:9])
                unrel = (vol(t * (v @ FLAT)) - 8) / t ** 4
                best = (r.fun, unrel, np.linalg.norm(r.x[9:]))
        print('%.2f   %.6f                   %.6f                 %.3f     [%.0fs]' % (t, best[0], best[1], best[2], time.time() - t0),
              flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
code25_search.py -- how much slack a twenty-fifth point needs: 25 points of S^3 with
small largest inner product, by descent on a smoothed maximum from random starts.

The theorem that the kissing number is stable excludes 25 points with inner products
at most 1/2 + 0.0065; this search bounds the true threshold from the other side.
It writes the best configuration found to runs/code25_best.txt and prints its
largest inner product, recomputed from the written coordinates.  Exploration:
floating point; the witness is a set of points, checked by recomputing its Gram
matrix.

Usage: python3 code25_search.py [seed] [starts]
"""
import os
import sys

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
N = 25
IU = np.triu_indices(N, 1)


def smooth_max(x, beta):
    X = x.reshape(N, 4)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    Y = X / n
    g = (Y @ Y.T)[IU]
    m = g.max()
    w = np.exp(beta * (g - m))
    s = w.sum()
    dG = np.zeros((N, N)); dG[IU] = w / s; dG = dG + dG.T
    dY = dG @ Y
    return m + np.log(s) / beta, ((dY - Y * (dY * Y).sum(1, keepdims=True)) / n).ravel()


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    starts = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    rng = np.random.default_rng(seed)
    best, bestX = 1.0, None
    for t in range(starts):
        x = rng.normal(size=N * 4)
        for beta in (20, 60, 200, 600, 2000, 6000):
            x = minimize(smooth_max, x, args=(beta,), jac=True, method='L-BFGS-B', options=dict(maxiter=3000)).x
        X = x.reshape(N, 4) / np.linalg.norm(x.reshape(N, 4), axis=1, keepdims=True)
        m = (X @ X.T)[IU].max()
        if m < best:
            best, bestX = m, X
        print('start %2d: largest inner product %.6f   best %.6f' % (t, m, best), flush=True)
    out = os.path.join(HERE, 'runs', 'code25_best.txt')
    np.savetxt(out, bestX, fmt='%.17f')
    Y = np.loadtxt(out)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    m = (Y @ Y.T)[IU].max()
    print('written to runs/code25_best.txt: 25 points, largest inner product %.6f, minimal angle %.3f degrees'
          % (m, np.degrees(np.arccos(m))))
    print('so a 25-point code exists with inner products at most 1/2 + %.4f' % (m - 0.5))


if __name__ == '__main__':
    main()

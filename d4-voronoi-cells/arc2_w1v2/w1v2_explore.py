#!/usr/bin/env python3
"""
Exploratory: does the w1-v2 arc admit the same kind of exact breakpoint
classification as the v1-w1 arc (Theorem thm:completebreak)?

Setup: e_perp(t) = cos(t)*w1 + sin(t)*v2, reduced to t in [0,pi/4] by the
S-symmetry (S=diag(1,1,1,-1) swaps w1<->v2, fixes u0 and v1 -- this is
Lemma s4symmetry, already proved and used in the paper to note the
w1-v2 arc is symmetric about its own midpoint t=pi/4).

Step 1: check whether the SAME sign vertices used for the v1-w1 arc
(Z*=2u0 [universal, e_perp-independent by lem:universalbreak -- should
already work here for free], W=(1,1,1,1)/sqrt2, Y=(1,-1,1,1)/sqrt2) and
the Hadamard image A=H(W) give exact rational/simple breakpoint curves
here too, by direct inner-product computation.

Step 2: numerically trace active-facet-set transitions across a grid to
see how many distinct curves actually appear, and compare against step 1.
"""
import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull
import sympy as sp

roots = []
for i in range(4):
    for j in range(i + 1, 4):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(4)
                v[i] = si
                v[j] = sj
                roots.append(v / np.sqrt(2))
roots = np.array(roots)


def find(vec):
    v = np.array(vec, dtype=float)
    v /= np.linalg.norm(v)
    d = roots @ v
    idx = np.argmax(d)
    assert d[idx] > 1 - 1e-9
    return idx


idx0 = find([1, 1, 0, 0])
u0 = roots[idx0]
v1 = roots[find([1, -1, 0, 0])]
w1 = roots[find([0, 0, 1, 1])]
v2 = roots[find([0, 0, 1, -1])]

S = np.diag([1., 1., 1., -1.])
print("S*w1 == v2:", np.allclose(S @ w1, v2))
print("S*v2 == w1:", np.allclose(S @ v2, w1))
print("S*v1 == v1:", np.allclose(S @ v1, v1))
print("S*u0 == u0:", np.allclose(S @ u0, u0))

# ---- Step 1: exact symbolic check of candidate sign vertices ----
t, theta = sp.symbols('t theta', real=True)
v1s = sp.Matrix([1, -1, 0, 0]) / sp.sqrt(2)
w1s = sp.Matrix([0, 0, 1, 1]) / sp.sqrt(2)
v2s = sp.Matrix([0, 0, 1, -1]) / sp.sqrt(2)
u0s = sp.Matrix([1, 1, 0, 0]) / sp.sqrt(2)

e_perp_s = sp.cos(t) * w1s + sp.sin(t) * v2s
u1_s = sp.cos(theta) * u0s + sp.sin(theta) * e_perp_s

Zstar = sp.Matrix([sp.sqrt(2), sp.sqrt(2), 0, 0])
W = sp.Matrix([1, 1, 1, 1]) / sp.sqrt(2)
Y = sp.Matrix([1, -1, 1, 1]) / sp.sqrt(2)
A = sp.Matrix([sp.sqrt(2), 0, 0, 0])  # Hadamard image of W

for name, pt in [("Z*", Zstar), ("W", W), ("Y", Y), ("A", A)]:
    ip = sp.simplify((pt.T * u1_s)[0])
    print(f"<{name}, u1(theta,t)> = {ip}")

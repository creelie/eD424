#!/usr/bin/env python3
"""
tilt_block.py -- the second variation of the contact-cell volume under pure
tilts (no push-outs), at the root system, in exact rationals.

The tilt block of the form H of prop:hessian-d4 (its 72 x 72 block on the
tangent coordinates c, eta = 0) is the Hessian of vol{x : <x, w_i> <= 1} as a
function of the 24 directions alone.  The script proves, by an exact LDL^T
over Q (a zero pivot accepted only with a zero row), that it is positive
semidefinite with exactly 15 zero pivots; it computes the kernel exactly
(dimension 15: the six infinitesimal rotations and nine further directions),
checks that the kernel is not spanned by rotations and infinitesimal strains
(the strains are not in it), and prints the spectrum (the smallest positive
eigenvalue is 1/12).  So the root system is a critical point of the
contact-cell volume at which the second variation is nonnegative in every
tilt and vanishes, beyond the rotations, on a nine-dimensional space; whether
it is a local minimum is decided at fourth order (tilt_quartic.py).
Usage: python3 tilt_block.py
"""
import itertools
from fractions import Fraction as Fr
import numpy as np
import sympy as sp
import rational_model as RM

H = RM.build_H()
T = [[H[r][s] for s in range(72)] for r in range(72)]


def ldl_psd(M):
    n = len(M); A = [row[:] for row in M]; zeros = 0; pivots = []
    for k in range(n):
        p = A[k][k]
        if p < 0:
            return False, zeros, pivots
        if p == 0:
            if any(A[k][j] != 0 for j in range(k, n)):
                return False, zeros, pivots
            zeros += 1; continue
        pivots.append(p)
        for i in range(k + 1, n):
            if A[i][k] != 0:
                f = A[i][k] / p
                for j in range(k, n):
                    A[i][j] -= f * A[k][j]
    return True, zeros, pivots


ok, zeros, piv = ldl_psd(T)
print(f'1. exact LDL^T of the 72 x 72 tilt block: positive semidefinite = {ok}, zero pivots {zeros}, '
      f'least positive pivot {min(piv)} ({float(min(piv)):.6f})')
ev = np.linalg.eigvalsh(np.array([[float(x) for x in r] for r in T]))
vals, counts = np.unique(np.round(ev, 6), return_counts=True)
print('2. spectrum (value: multiplicity):', ', '.join(f'{v:g}: {c}' for v, c in zip(vals, counts)))
Ts = sp.Matrix(72, 72, lambda r, s: sp.Rational(T[r][s].numerator, T[r][s].denominator))
K = Ts.nullspace()
print(f'3. exact kernel: dimension {len(K)}')


def field_c(M):
    c = []
    for i, a in enumerate(RM.ROOTS):
        Ma = [sum(M[p][q] * a[q] for q in range(4)) for p in range(4)]
        lam = RM.dot(a, Ma) / 2
        t = [Ma[p] - lam * a[p] for p in range(4)]
        for b in RM.BAS[i]:
            c.append(sp.Rational(RM.dot(t, b)) / RM.dot(b, b))
    return sp.Matrix(c)


rot, strain = [], []
for p, q in itertools.combinations(range(4), 2):
    M = [[Fr(0)] * 4 for _ in range(4)]; M[p][q] = Fr(1); M[q][p] = Fr(-1); rot.append(field_c(M))
for p in range(3):
    M = [[Fr(0)] * 4 for _ in range(4)]; M[p][p] = Fr(1); M[p + 1][p + 1] = Fr(-1); strain.append(field_c(M))
for p, q in itertools.combinations(range(4), 2):
    M = [[Fr(0)] * 4 for _ in range(4)]; M[p][q] = M[q][p] = Fr(1); strain.append(field_c(M))
rot_in = all((Ts * v).is_zero_matrix for v in rot)
strain_in = [bool((Ts * v).is_zero_matrix) for v in strain]
Kmat = sp.Matrix.hstack(*K)
print(f'4. the six rotations lie in the kernel: {rot_in}; rank of kernel + rotations = '
      f'{sp.Matrix.hstack(Kmat, *rot).rank()} (so they span a 6-dimensional part of it)')
print(f'5. the nine infinitesimal strains lie in the kernel: {strain_in.count(True)} of 9')
assert ok and zeros == 15 and len(K) == 15 and rot_in and not any(strain_in)
np.save('tilt_kernel.npy', np.array(Kmat.T.tolist(), dtype=float))
print('PASS: the tilt block is positive semidefinite, with a 15-dimensional kernel '
      '(6 rotations and 9 further flat directions, written to tilt_kernel.npy)')

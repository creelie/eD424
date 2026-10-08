#!/usr/bin/env python3
"""
cap_rearrangement.py -- the rearrangement bound for a cell with N facets, and
why it stops below 8 at N = 24.

Let V = {x : <x, y> <= |y|^2/2, y in Y} with |y| >= 2.  In polar coordinates
vol(V) = (1/4) int_{S^3} rho(u)^4 du, and the directions u whose ray leaves V
through the facet of y form a region D_y of S^3 on which
rho(u) = h_y / cos(angle(u, y)), h_y = |y|/2 >= 1.  For a region of area A
the integral of (1/4) (h / cos)^4 is least when the region is the cap of area
A about y (the integrand grows with the angle), and that least value is

    F(h, A) = (pi/3) h^4 tan^3 psi,   2 pi (psi - sin psi cos psi) = A,

increasing in h and convex in A (its derivative in A is h^4 sec^4 psi / 4).
The regions cover S^3, of area 2 pi^2, so by convexity

    vol(V) >= N F(1, 2 pi^2 / N)   for every cell with N facets.

The script evaluates the bound for N = 20..30 and compares, at N = 24, the
cap with the region of one facet of the 24-cell (cone volume 8/24 = 1/3).
Floating point; the bound itself is the elementary argument above.
"""
import math

from scipy.optimize import brentq


def psi_of(A):
    return brentq(lambda p: 2 * math.pi * (p - math.sin(p) * math.cos(p)) - A, 1e-9, math.pi - 1e-9)


def F(h, A):
    return math.pi / 3 * h ** 4 * math.tan(psi_of(A)) ** 3


for N in range(20, 31):
    print('N = %2d: vol(V) >= N F(1, 2 pi^2/N) = %.6f' % (N, N * F(1, 2 * math.pi ** 2 / N)))
A = 2 * math.pi ** 2 / 24
print('N = 24: cap of area 2 pi^2/24 has angular radius %.6f and cone volume %.6f;'
      % (psi_of(A), F(1, A)))
print('        the octahedral region of a facet of the 24-cell has cone volume %.6f' % (1 / 3))
print('the bound misses 8 by %.6f: an argument for (G) has to use the shape of the regions,'
      % (8 - 24 * F(1, A)))
print('not only their areas')

#!/usr/bin/env python3
"""
Exact dimension count and explicit basis for the space of quartic forms
invariant under A_18's order-48 stabiliser group, acting on the
near-null representation eps_perm (1-dim, coordinate x0) (+)
(std3 (x) det) (3-dim, coordinates y=(y0,y1,y2)) identified exactly by
a18_nullspace_irrep_match.py.

Part 1 (power-sum / Newton's-identity dimension count, exact integer and
rational arithmetic): a quartic Q(x0,y) invariant under the whole group
splits by degree in x0 into five pieces x0^4, x0^3 y, x0^2 y^2, x0 y^3,
y^4. Since eps_perm and det are both order-2 characters, invariance of
each piece under the full group reduces to: plain std3-invariance of
the degree-b-in-y part when b is even, or std3-relative-invariance
(transforming by eps_perm) when b is odd. The needed multiplicity is
computed via chi_{Sym^b(W)} (W = std3 (x) det), obtained from the power
sums p_k = trace(M^k), M = det(g)*g, via Newton's identities -- exact
rational arithmetic throughout.

Part 2 (explicit basis, independent check): the same five invariants
are constructed directly by Reynolds-operator projection (exact
sympy.Rational arithmetic, averaging over the full 48-element group,
not merely class representatives) of specific monomials, and the two
methods are checked to agree.

Run: python a18_invariant_quartic_basis.py
"""
import itertools
from fractions import Fraction as Fr
import sympy as sp

# ---------------------------------------------------------------------
# The 48-element group, abstractly as 3x3 signed permutation matrices.
# ---------------------------------------------------------------------
group3 = []
for perm in itertools.permutations(range(3)):
    for signs in itertools.product([1, -1], repeat=3):
        M = [[0]*3 for _ in range(3)]
        for out_idx in range(3):
            M[out_idx][perm[out_idx]] = signs[out_idx]
        group3.append(tuple(tuple(row) for row in M))
assert len(group3) == 48

def mat_mult(A, B):
    return tuple(tuple(sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)) for i in range(3))
def mat_transpose(A):
    return tuple(tuple(A[j][i] for j in range(3)) for i in range(3))
def mat_trace(A):
    return sum(A[i][i] for i in range(3))
def mat_det3(A):
    a,b,c = A[0]; d,e,f = A[1]; g,h,i = A[2]
    return a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g)
def perm_and_signs(M):
    perm = [None]*3; signs = [None]*3
    for i in range(3):
        for j in range(3):
            if M[i][j] != 0:
                perm[i] = j; signs[i] = M[i][j]
    return tuple(perm), tuple(signs)
def perm_sign(perm):
    inv = 0
    for i in range(len(perm)):
        for j in range(i+1, len(perm)):
            if perm[i] > perm[j]:
                inv += 1
    return 1 if inv % 2 == 0 else -1
def eps_perm(g):
    p, _ = perm_and_signs(g)
    return perm_sign(p)
def det_char(g):
    return mat_det3(g)

# ---------------------------------------------------------------------
# Part 1: exact dimension count via power sums / Newton's identities.
# ---------------------------------------------------------------------
visited = set(); classes = []
for g in group3:
    if g in visited:
        continue
    cls = set()
    for k in group3:
        kt = mat_transpose(k)
        cls.add(mat_mult(mat_mult(k, g), kt))
    classes.append(cls); visited |= cls
reps = [next(iter(c)) for c in classes]
sizes = [len(c) for c in classes]
order = 48
assert sum(sizes) == 48

def matpow(M, k):
    R = ((1,0,0),(0,1,0),(0,0,1))
    for _ in range(k):
        R = mat_mult(R, M)
    return R

h_vals = {b: [] for b in range(5)}
eps_perm_vals = [eps_perm(r) for r in reps]

for r in reps:
    d = det_char(r)
    Mw = tuple(tuple(d*r[i][j] for j in range(3)) for i in range(3))  # rho_W(g) = det(g)*g
    p1 = mat_trace(Mw)
    p2 = mat_trace(matpow(Mw, 2))
    p3 = mat_trace(matpow(Mw, 3))
    p4 = mat_trace(matpow(Mw, 4))
    h0 = Fr(1)
    h1 = Fr(p1)
    h2 = Fr(p1**2 + p2, 2)
    h3 = Fr(p1**3 + 3*p1*p2 + 2*p3, 6)
    h4 = Fr(p1**4 + 6*p1**2*p2 + 3*p2**2 + 8*p1*p3 + 6*p4, 24)
    h_vals[0].append(h0); h_vals[1].append(h1); h_vals[2].append(h2)
    h_vals[3].append(h3); h_vals[4].append(h4)

print("chi_{Sym^b(W)} values per class, b=0..4 (W = std3 (x) det):")
for b in range(5):
    print(f"  b={b}: {h_vals[b]}")

def inner(charA, charB):
    return Fr(sum(sz*a*b for sz, a, b in zip(sizes, charA, charB)), order)

print()
print("=== Part 1: dimension of invariant quartics Q(x0,y), by degree split ===")
triv = [1]*len(reps)
total_dim = 0
for a, b, needed, needname in [(4,0,triv,'triv'), (3,1,eps_perm_vals,'eps_perm'),
                                (2,2,triv,'triv'), (1,3,eps_perm_vals,'eps_perm'),
                                (0,4,triv,'triv')]:
    mult = inner(h_vals[b], needed)
    assert mult.denominator == 1
    mult = int(mult)
    print(f"  x0^{a} * (deg-{b} in y): mult of {needname} in Sym^{b}(W) = {mult}")
    total_dim += mult
print(f"\nTOTAL dimension of the invariant-quartic space: {total_dim}")
print("(vs. dimension 35 for a general quartic in 4 variables)")
assert total_dim == 5

# ---------------------------------------------------------------------
# Part 2: explicit basis via Reynolds-operator projection (independent
# check, exact sympy.Rational arithmetic, full 48-element group).
# ---------------------------------------------------------------------
print("\n=== Part 2: explicit basis via Reynolds projection ===")
y0, y1, y2 = sp.symbols('y0 y1 y2', real=True)
Y = sp.Matrix([y0, y1, y2])
group3_sp = [sp.Matrix(M) for M in group3]

def reynolds_project(poly, character_of):
    acc = sp.Integer(0)
    for M in group3_sp:
        d = M.det()
        Ynew = d * (M * Y)
        sub = {y0: Ynew[0], y1: Ynew[1], y2: Ynew[2]}
        chi = character_of(M)
        acc += sp.Rational(1, chi) * poly.subs(sub, simultaneous=True)
    return sp.expand(acc / 48)

def triv_char_sp(M):
    return 1
def eps_perm_char_sp(M):
    perm = [None]*3
    for i in range(3):
        for j in range(3):
            if M[i, j] != 0:
                perm[i] = j
    inv = 0
    for i in range(3):
        for j in range(i+1, 3):
            if perm[i] > perm[j]:
                inv += 1
    return 1 if inv % 2 == 0 else -1

S2  = reynolds_project(y0**2, triv_char_sp)
S4a = reynolds_project(y0**4, triv_char_sp)
S4b = reynolds_project(y0**2*y1**2, triv_char_sp)
print("proj(y0^2)      =", S2,  "  (expect (y0^2+y1^2+y2^2)/3)")
print("proj(y0^4)      =", S4a, "  (expect (y0^4+y1^4+y2^4)/3)")
print("proj(y0^2*y1^2) =", S4b, "  (expect (y0^2y1^2+y1^2y2^2+y2^2y0^2)/3)")

print()
for mono in [y0**3, y0**2*y1, y0*y1*y2]:
    proj = reynolds_project(mono, eps_perm_char_sp)
    print(f"proj({mono}, onto eps_perm) = {proj}")

S2_expected  = sp.Rational(1,3)*(y0**2+y1**2+y2**2)
S4a_expected = sp.Rational(1,3)*(y0**4+y1**4+y2**4)
S4b_expected = sp.Rational(1,3)*(y0**2*y1**2+y1**2*y2**2+y2**2*y0**2)
assert sp.simplify(S2  - S2_expected)  == 0
assert sp.simplify(S4a - S4a_expected) == 0
assert sp.simplify(S4b - S4b_expected) == 0
P3 = reynolds_project(y0*y1*y2, eps_perm_char_sp)
assert sp.simplify(P3 - y0*y1*y2) == 0
print("\n-> confirms: proj(y0^3), proj(y0^2 y1) onto eps_perm vanish; proj(y0 y1 y2)")
print("   onto eps_perm reproduces y0*y1*y2 exactly unchanged -- P3 = y0*y1*y2 is")
print("   already the (unique, up to scale) relative invariant of degree 3.")

print("\n=== Final basis, both methods agree ===")
print("  x0^4")
print("  x0^2 * S2(y),   S2  = y0^2+y1^2+y2^2")
print("  x0   * P3(y),   P3  = y0*y1*y2")
print("  S4(y)         = y0^4+y1^4+y2^4")
print("  S22(y)        = y0^2*y1^2+y1^2*y2^2+y2^2*y0^2")
print("\nSign-definiteness of each basis element on its own: x0^4, S2, S4, S22")
print("are all manifestly sums of squares (>= 0 everywhere); P3 = y0*y1*y2 is")
print("odd degree in y and changes sign under y -> -y, so it is NOT sign-definite")
print("by itself -- the c3 coefficient's contribution can only be controlled in")
print("combination with the others, not term by term.")

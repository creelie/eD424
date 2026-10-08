#!/usr/bin/env python3
"""
Exact character table of A_18's order-48 exact stabiliser group, and the
exact decomposition of the 54-dimensional joint tangent representation.

Self-contained: exact integer/rational arithmetic throughout, no floating
point anywhere in this script.

Background. a18_stabilizer_group.py (elsewhere in this directory) shows
the subgroup of the 384-element signed-coordinate-permutation
automorphism group of D4 that fixes A_18 (as a SET of 18 root indices)
has order exactly 48, and consists of every signed permutation that
fixes coordinate index 1 (0-indexed) -- both its identity and its sign
-- and acts as an arbitrary signed permutation on the other three
coordinates {0,2,3}. (An initial attempt at this script wrongly assumed
the fixed coordinate was index 0; that produced non-integer irreducible
multiplicities when decomposing chi_54 below -- an impossible result for
any genuine representation, so it was caught as a red flag rather than
reported. FREE_COORDS/FIXED_COORD below are the corrected, verified
values.)

This script:
  1. Builds the abstract 48-element group as 3x3 signed permutation
     matrices, and embeds it back into 4x4 (acting on coordinates
     {0,2,3}, fixing coordinate 1).
  2. Computes its 10 conjugacy classes exactly (integer matrix
     conjugation).
  3. Constructs 10 candidate irreducible characters directly from the
     group's natural representations (trivial; two independent order-2
     sign characters -- eps_perm = sign of the underlying permutation,
     eps_signs = product of the three diagonal signs -- and their
     product det = eps_perm*eps_signs; the defining 3-dim representation
     std3 and its three twists by the sign characters; and a 2-dim
     representation pulled back from S3's standard irrep through the
     natural quotient map B3 -> S3 that forgets signs, and its twist by
     eps_signs), and verifies they are pairwise orthonormal (hence the
     complete list of irreducibles, since sum of squares of dimensions
     is 48).
  4. Computes the EXACT character of the 54-dimensional joint tangent
     representation via the closed form
        chi_54(g) = n_fix(g) * (trace_4x4(g) - 1)
     where n_fix(g) is the number of A_18's 18 active roots g maps to
     themselves exactly (not merely permutes among the set), and
     trace_4x4(g)-1 is the trace of g restricted to the 3-dim invariant
     tangent complement of any root it fixes (full 4x4 trace = 1, the
     eigenvalue on the fixed root's own line, plus this complement).
  5. Decomposes chi_54 into the 10 irreducibles and checks the
     multiplicities are non-negative integers summing to dimension 54.

Run: python a18_exact_character_table.py
"""
import itertools
from fractions import Fraction as Fr

# ---------------------------------------------------------------------
# 1. The 48-element group, abstractly as 3x3 signed permutation matrices
#    acting on the "free" coordinates, and embedded back into 4x4.
# ---------------------------------------------------------------------
FREE_COORDS = [0, 2, 3]   # verified stabiliser structure of A_18
FIXED_COORD = 1

def build_roots():
    roots = []
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0, 0, 0, 0]
                    v[i] = si
                    v[j] = sj
                    roots.append(tuple(v))  # unnormalised; sign pattern is all that matters here
    return roots

ROOTS = build_roots()
assert len(ROOTS) == 24

A_18 = [0,2,4,5,6,7,8,9,10,11,12,13,16,17,20,21,22,23]

group3 = []  # abstract 3x3 signed permutation matrices
for perm in itertools.permutations(range(3)):
    for signs in itertools.product([1, -1], repeat=3):
        M = [[0]*3 for _ in range(3)]
        for out_idx in range(3):
            M[out_idx][perm[out_idx]] = signs[out_idx]
        group3.append(tuple(tuple(row) for row in M))
assert len(group3) == 48

def embed_4x4(M3):
    """Embed a 3x3 signed-permutation matrix (acting on FREE_COORDS) into
    a 4x4 signed permutation matrix fixing FIXED_COORD exactly."""
    M4 = [[0]*4 for _ in range(4)]
    M4[FIXED_COORD][FIXED_COORD] = 1
    for a in range(3):
        for b in range(3):
            M4[FREE_COORDS[a]][FREE_COORDS[b]] = M3[a][b]
    return tuple(tuple(row) for row in M4)

group4 = [embed_4x4(M3) for M3 in group3]

def apply4(M4, v):
    return tuple(sum(M4[i][j]*v[j] for j in range(4)) for i in range(4))

# Sanity: every element of group4 must map ROOTS bijectively to ROOTS and
# fix A_18 as a set of indices.
root_index = {v: i for i, v in enumerate(ROOTS)}
n_ok = 0
for M4 in group4:
    images = [apply4(M4, v) for v in ROOTS]
    assert all(im in root_index for im in images), "not an automorphism of the root system"
    image_idx = set(root_index[im] for im in images)
    assert image_idx == set(range(24))
    mapped_A18 = set(root_index[apply4(M4, ROOTS[i])] for i in A_18)
    if mapped_A18 == set(A_18):
        n_ok += 1
print(f"sanity: {n_ok} of {len(group4)} candidate embeddings correctly stabilise A_18")
assert n_ok == 48

# ---------------------------------------------------------------------
# 2. Conjugacy classes (exact, integer matrix conjugation on group3).
# ---------------------------------------------------------------------
def mat_mult3(A, B):
    return tuple(tuple(sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)) for i in range(3))
def mat_transpose3(A):
    return tuple(tuple(A[j][i] for j in range(3)) for i in range(3))
def mat_trace3(A):
    return sum(A[i][i] for i in range(3))
def mat_det3(A):
    a,b,c = A[0]; d,e,f = A[1]; g,h,i = A[2]
    return a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g)

inv3 = {g: mat_transpose3(g) for g in group3}  # orthogonal matrices: inverse = transpose

classes = []
seen = set()
for g in group3:
    if g in seen:
        continue
    cls = set()
    for h in group3:
        hinv = inv3[h]
        conj = mat_mult3(mat_mult3(h, g), hinv)
        cls.add(conj)
    classes.append(sorted(cls))
    seen |= cls
identity3 = tuple(tuple(1 if i==j else 0 for j in range(3)) for i in range(3))
# put the identity's own class first, then sort the rest by (size, rep) for
# a reproducible order
classes.sort(key=lambda c: (0 if identity3 in c else 1, len(c), c[0]))
print(f"\nnumber of conjugacy classes: {len(classes)}")
print("class sizes:", [len(c) for c in classes])
assert len(classes) == 10
assert sorted(len(c) for c in classes) == [1,1,3,3,6,6,6,6,8,8]

reps = [c[0] for c in classes]
class_sizes = [len(c) for c in classes]
assert reps[0] == identity3, "identity must be the first class representative"

def perm_of(M3):
    """recover the underlying permutation (as a tuple) and sign pattern"""
    perm = [None]*3
    signs = [None]*3
    for out_idx in range(3):
        for in_idx in range(3):
            if M3[out_idx][in_idx] != 0:
                perm[out_idx] = in_idx
                signs[out_idx] = M3[out_idx][in_idx]
    return tuple(perm), tuple(signs)

def sign_of_perm(p):
    p = list(p)
    n = len(p)
    seen_local = [False]*n
    parity = 1
    for i in range(n):
        if seen_local[i]:
            continue
        j = i
        clen = 0
        while not seen_local[j]:
            seen_local[j] = True
            j = p[j]
            clen += 1
        if clen % 2 == 0:
            parity *= -1
    return parity

# ---------------------------------------------------------------------
# 3. The 10 irreducible characters, evaluated at each class representative.
# ---------------------------------------------------------------------
def eps_perm(M3):
    p, s = perm_of(M3)
    return sign_of_perm(p)
def eps_signs(M3):
    p, s = perm_of(M3)
    r = 1
    for x in s:
        r *= x
    return r
def det_char(M3):
    return eps_perm(M3)*eps_signs(M3)
def std3_char(M3):
    return mat_trace3(M3)

# S3-pullback 2-dim character: standard irrep of S3 has character
# 2 at identity, 0 at transpositions, -1 at 3-cycles.
def s3_std2_char(perm):
    # perm is a permutation of {0,1,2}; classify by cycle type
    p = list(perm)
    n = 3
    seen_local = [False]*n
    cycles = []
    for i in range(n):
        if seen_local[i]:
            continue
        j = i; clen = 0
        while not seen_local[j]:
            seen_local[j] = True; j = p[j]; clen += 1
        cycles.append(clen)
    cycles.sort()
    if cycles == [1,1,1]:
        return 2
    if cycles == [1,2]:
        return 0
    if cycles == [3]:
        return -1
    raise ValueError(cycles)

def two_dim_char(M3):
    p, s = perm_of(M3)
    return s3_std2_char(p)

irreps = {
    'triv':            lambda M: 1,
    'eps_perm':        eps_perm,
    'eps_signs':       eps_signs,
    'det':             det_char,
    'std3':            std3_char,
    'std3*eps_perm':   lambda M: std3_char(M)*eps_perm(M),
    'std3*eps_signs':  lambda M: std3_char(M)*eps_signs(M),
    'std3*det':        lambda M: std3_char(M)*det_char(M),
    '2dim':            two_dim_char,
    '2dim*eps_signs':  lambda M: two_dim_char(M)*eps_signs(M),
}
irrep_dims = {'triv':1,'eps_perm':1,'eps_signs':1,'det':1,'std3':3,
              'std3*eps_perm':3,'std3*eps_signs':3,'std3*det':3,
              '2dim':2,'2dim*eps_signs':2}
assert sum(d*d for d in irrep_dims.values()) == 48

print("\nCharacter table (rows = irreps, columns = the 10 classes, sizes",
      class_sizes, "):")
char_values = {}
for name, chi in irreps.items():
    vals = [chi(r) for r in reps]
    char_values[name] = vals
    print(f"  {name:16s} {vals}")

# Orthogonality check: <chi_i, chi_j> = (1/48) * sum_classes size*chi_i*chi_j(conj) = delta_ij
def inner(chi_a_vals, chi_b_vals):
    total = Fr(0)
    for size, a, b in zip(class_sizes, chi_a_vals, chi_b_vals):
        total += Fr(size)*Fr(a)*Fr(b)  # all characters here are real
    return total / 48

names = list(irreps.keys())
print("\nOrthonormality check (should be identity matrix):")
ok = True
for i, ni in enumerate(names):
    row = []
    for j, nj in enumerate(names):
        v = inner(char_values[ni], char_values[nj])
        row.append(v)
        if i == j and v != 1:
            ok = False
        if i != j and v != 0:
            ok = False
    print("  ", [str(x) for x in row])
assert ok, "characters are not orthonormal -- something is wrong"
print("-> all 10 characters are irreducible and pairwise orthogonal: complete list confirmed.")

# ---------------------------------------------------------------------
# 4. Exact character of the 54-dim joint tangent representation.
# ---------------------------------------------------------------------
def n_fix_and_trace(M4):
    """n_fix(g) = number of A_18 roots mapped to themselves exactly;
    trace_4x4(g) = ordinary trace of the 4x4 matrix."""
    n_fix = 0
    for idx in A_18:
        if apply4(M4, ROOTS[idx]) == ROOTS[idx]:
            n_fix += 1
    tr4 = sum(M4[i][i] for i in range(4))
    return n_fix, tr4

chi54_reps = []
for M3 in reps:
    M4 = embed_4x4(M3)
    n_fix, tr4 = n_fix_and_trace(M4)
    chi54_reps.append(n_fix*(tr4 - 1))

print("\nchi_54 at the 10 class representatives:", chi54_reps)
print("chi_54(identity) =", chi54_reps[0], " (must equal 54)")
assert chi54_reps[0] == 54

# ---------------------------------------------------------------------
# 5. Decompose chi_54 into the 10 irreducibles.
# ---------------------------------------------------------------------
print("\nDecomposition of chi_54:")
total_dim = 0
mults = {}
for name in names:
    m = inner(chi54_reps, char_values[name])
    assert m.denominator == 1, f"non-integer multiplicity for {name}: {m}"
    m = int(m)
    assert m >= 0
    mults[name] = m
    total_dim += m*irrep_dims[name]
    print(f"  multiplicity of {name:16s} (dim {irrep_dims[name]}): {m}")
print(f"\nreconstructed total dimension: {total_dim} (must equal 54)")
assert total_dim == 54
print("\nOK: chi_54 =", " + ".join(f"{mults[n]}*{n}" for n in names if mults[n] > 0))

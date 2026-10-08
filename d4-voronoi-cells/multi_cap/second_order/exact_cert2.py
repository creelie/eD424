"""
Exact copositivity certificate for  H + c c^T  on the cone  {B xi >= 0}:

    H + c c^T  =  P + B^T N B,     P psd (exact LDL^T),  N >= 0 entrywise, rational.

Stage 1: kernel of the tight certificate.  Tight directions xi* (H(xi*,xi*) = -(c xi*)^2,
         xi* in the cone) satisfy xi*^T P xi* = 0, hence P xi* = 0.  The 24 one-centre
         rays span the whole eta-space, so P_{eta *} = 0; the 6 rotations are in ker P;
         and any further tight tau-directions are found numerically and identified.
Stage 2: N as a combination of orbit indicators of W(B4) acting on the 120 constraints
         (the numerical certificate averaged over the group has this form).
Stage 3: impose P k = 0 exactly for every kernel vector k (linear in the orbit values),
         solve over Q, choose the free parameters nearest to the numerical values,
         check N >= 0 and P psd by exact LDL^T.
"""
import itertools, sys
from fractions import Fraction as Fr
import numpy as np
sys.path.insert(0, '.')
import rational_model as RM

N = 96
H = RM.build_H(); B = RM.build_B(); c = RM.cvec()
Hf = RM.to_float(H); Bf = RM.to_float(B); cf = np.array([float(x) for x in c])
M = [[H[p][q] + c[p] * c[q] for q in range(N)] for p in range(N)]
Mf = Hf + np.outer(cf, cf)
Navg = np.load('N_avg.npy')

# ---------------- stage 1: kernel of P (numerically, in the rational coordinates)
Pf = Mf - Bf.T @ Navg @ Bf
Pf = (Pf + Pf.T) / 2
w, V = np.linalg.eigh(Pf)
print("eigenvalues of P (numerical certificate, rational coordinates):")
print("   smallest 40:", np.round(w[:40], 6))
kdim = int((np.abs(w) < 1e-5).sum())
print(f"   numerical kernel dimension: {kdim}   (24 eta + 6 rotations = 30 expected at least)")

# exact kernel candidates: eta directions and rotations
def rotation_vectors():
    """infinitesimal rotations A (antisymmetric): tau_i = A u_i, so t_i = A a_i; coordinates c_ik = <t_i, b_ik>/|b_ik|^2"""
    vecs = []
    for p, q in itertools.combinations(range(4), 2):
        A = [[Fr(0)] * 4 for _ in range(4)]
        A[p][q] = Fr(1); A[q][p] = Fr(-1)
        xi = [Fr(0)] * N
        for i in range(24):
            a = RM.ROOTS[i]
            t = tuple(sum(A[r][s] * a[s] for s in range(4)) for r in range(4))
            for k in range(3):
                xi[RM.idx_c(i, k)] = RM.dot(t, RM.BAS[i][k]) / RM.NORM2[i][k]
        vecs.append(xi)
    return vecs

ROT = rotation_vectors()
ETA = []
for i in range(24):
    xi = [Fr(0)] * N; xi[RM.idx_eta(i)] = Fr(1); ETA.append(xi)
# check that these are (numerically) in ker P and that they exhaust it
K = np.array([[float(x) for x in v] for v in ROT + ETA])
resid = np.abs(Pf @ K.T).max()
print(f"   |P k| for the 30 exact kernel candidates: {resid:.2e}")
# projector onto their complement; residual spectrum
Q, _ = np.linalg.qr(K.T)
Pc = np.eye(N) - Q @ Q.T
wc = np.linalg.eigvalsh(Pc @ Pf @ Pc)
print("   spectrum of P on the complement of those 30 (smallest 12):", np.round(np.sort(wc)[:12], 6))
extra = int((np.abs(np.sort(wc)) < 1e-5).sum()) - 30
print(f"   further near-zero directions beyond the 30: {extra}")

# ---------------- stage 2: orbits of constraint pairs under W(B4)
def signed_perms():
    for perm in itertools.permutations(range(4)):
        for signs in itertools.product((1, -1), repeat=4):
            yield perm, signs

root_idx = {r: i for i, r in enumerate(RM.ROOTS)}
pair_idx = {p: r for r, p in enumerate(RM.TIGHT)}
perms120 = []
for perm, signs in signed_perms():
    def act(a):
        out = [Fr(0)] * 4
        for r in range(4):
            out[r] = signs[r] * a[perm[r]]
        return tuple(out)
    sigma = [root_idx[act(a)] for a in RM.ROOTS]
    q = [0] * 120
    for r, (i, j) in enumerate(RM.TIGHT):
        a, b = sorted((sigma[i], sigma[j]))
        q[r] = pair_idx[(a, b)]
    for i in range(24):
        q[96 + i] = 96 + sigma[i]
    perms120.append(q)
print(f"\ngroup: {len(perms120)} signed permutations acting on the 120 constraints")

orbit_of = {}
orbits = []
for r in range(120):
    for s in range(r, 120):
        if (r, s) in orbit_of:
            continue
        orb = set()
        for q in perms120:
            a, b = q[r], q[s]
            orb.add((min(a, b), max(a, b)))
        oid = len(orbits); orbits.append(sorted(orb))
        for e in orb:
            orbit_of[e] = oid
print(f"orbits of unordered constraint pairs: {len(orbits)}")
nu_num = np.array([np.mean([Navg[a, b] for a, b in orb]) for orb in orbits])
spread = max(np.std([Navg[a, b] for a, b in orb]) for orb in orbits)
print(f"numerical orbit values (max within-orbit spread {spread:.1e}):")
print("  ", np.round(nu_num, 6))

# ---------------- stage 3: exact linear conditions  P k = 0  for the 30 kernel vectors
def BtEB_k(orb, k):
    """(B^T E_o B) k  where E_o = sum over (a,b) in orbit of (e_a e_b^T + e_b e_a^T) (halved on the diagonal)"""
    Bk = [sum(B[r][p] * k[p] for p in range(N)) for r in range(120)]   # (B k)_r
    coef = [Fr(0)] * 120                                               # (E_o B k)_r
    for a, b in orb:
        if a == b:
            coef[a] += Bk[a]
        else:
            coef[a] += Bk[b]; coef[b] += Bk[a]
    out = [sum(B[r][p] * coef[r] for r in range(120) if coef[r] != 0) for p in range(N)]
    return out

def Mk(k):
    return [sum(M[p][q] * k[q] for q in range(N) if k[q] != 0) for p in range(N)]

rows = []; rhs = []
for k in ROT + ETA:
    mk = Mk(k)
    cols = [BtEB_k(orb, k) for orb in orbits]
    for p in range(N):
        row = [cols[o][p] for o in range(len(orbits))]
        if any(x != 0 for x in row) or mk[p] != 0:
            rows.append(row); rhs.append(mk[p])
print(f"\nlinear conditions P k = 0: {len(rows)} equations in {len(orbits)} unknowns (before reduction)")

# exact Gaussian elimination (row reduce augmented system)
def rref(rows, rhs):
    m = len(rows); n = len(rows[0])
    A = [r[:] + [b] for r, b in zip(rows, rhs)]
    piv_cols = []; r0 = 0
    for col in range(n):
        piv = None
        for i in range(r0, m):
            if A[i][col] != 0:
                piv = i; break
        if piv is None:
            continue
        A[r0], A[piv] = A[piv], A[r0]
        pv = A[r0][col]
        A[r0] = [x / pv for x in A[r0]]
        for i in range(m):
            if i != r0 and A[i][col] != 0:
                f = A[i][col]
                A[i] = [x - f * y for x, y in zip(A[i], A[r0])]
        piv_cols.append(col); r0 += 1
        if r0 == m:
            break
    # consistency: rows beyond rank must be all zero
    for i in range(r0, m):
        if any(x != 0 for x in A[i]):
            return None, piv_cols
    return A[:r0], piv_cols

R, piv = rref(rows, rhs)
if R is None:
    print("INCONSISTENT: no certificate with this orbit structure satisfies P k = 0 exactly")
    sys.exit(1)
free = [o for o in range(len(orbits)) if o not in piv]
print(f"rank {len(piv)}; free orbit parameters: {len(free)}")

# choose free parameters = numerical values rounded to rationals (denominator 2^24), solve for the rest
den = 2 ** 24
nu = [None] * len(orbits)
for o in free:
    nu[o] = Fr(round(max(nu_num[o], 0) * den), den)
for row in R:
    col = next(i for i in range(len(orbits)) if row[i] != 0)     # pivot column (leading 1)
    val = row[-1] - sum(row[o] * nu[o] for o in free if row[o] != 0)
    nu[col] = val
negs = [o for o in range(len(orbits)) if nu[o] < 0]
print(f"exact orbit values: min = {float(min(nu)):.3e}; negative ones: {len(negs)}")
if negs:
    print("   negative values:", [(o, float(nu[o])) for o in negs])

# assemble N exactly and P = M - B^T N B
Nex = [[Fr(0)] * 120 for _ in range(120)]
for o, orb in enumerate(orbits):
    for a, b in orb:
        Nex[a][b] = nu[o]; Nex[b][a] = nu[o]
NB = [[sum(Nex[r][s] * B[s][p] for s in range(120) if Nex[r][s] != 0) for p in range(N)] for r in range(120)]
P = [[M[p][q] - sum(B[r][p] * NB[r][q] for r in range(120) if B[r][p] != 0) for q in range(N)] for p in range(N)]
# check P k = 0 exactly
ok = all(all(sum(P[p][q] * k[q] for q in range(N)) == 0 for p in range(N)) for k in ROT + ETA)
print("P k = 0 exactly for all 30 kernel vectors:", ok)
Pf2 = RM.to_float(P)
print("float eigenvalues of exact P (smallest 34):", np.round(np.linalg.eigvalsh(Pf2)[:34], 6))

# exact psd check: LDL^T with zero pivots allowed only with zero rows
def ldl_psd(Mx):
    n = len(Mx); A = [row[:] for row in Mx]
    for k in range(n):
        piv = A[k][k]
        if piv < 0:
            return False, k
        if piv == 0:
            if any(A[k][j] != 0 for j in range(k + 1, n)):
                return False, k
            continue
        for i in range(k + 1, n):
            if A[i][k] != 0:
                m = A[i][k] / piv
                for j in range(k + 1, n):
                    A[i][j] -= m * A[k][j]
    return True, None

psd, where = ldl_psd(P)
print("exact LDL^T: P positive semidefinite =", psd, "" if psd else f"(failed at pivot {where})")
if psd and not negs:
    print("\nCERTIFIED EXACTLY:  H + c c^T = P + B^T N B  with P >= 0 and N >= 0.  Hence m = -1.")
import pickle
pickle.dump({'nu': nu, 'orbits': orbits}, open('exact_certificate.pkl', 'wb'))

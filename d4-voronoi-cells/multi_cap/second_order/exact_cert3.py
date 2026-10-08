"""Final exact certificate: orbits with numerical value ~0 fixed at exactly 0, rest solved over Q."""
import sys, pickle
from fractions import Fraction as Fr
import numpy as np
sys.path.insert(0, '.')
exec(open('exact_cert2.py').read().split("# choose free parameters")[0])   # stages 1-2 and the rref helper
zero_orbits = [o for o in range(len(orbits)) if nu_num[o] < 1e-7]
unk = [o for o in range(len(orbits)) if o not in zero_orbits]
print(f"\norbits fixed at 0: {len(zero_orbits)}; unknown orbit values: {len(unk)}")
rows2 = [[row[o] for o in unk] for row in rows]
R, piv = rref(rows2, rhs)
assert R is not None, "inconsistent after fixing zero orbits"
free = [i for i in range(len(unk)) if i not in piv]
print(f"rank {len(piv)}; free parameters {len(free)}")
den = 2 ** 24
nu = [Fr(0)] * len(orbits)
val = [None] * len(unk)
for i in free:
    val[i] = Fr(round(nu_num[unk[i]] * den), den)
for row in R:
    col = next(i for i in range(len(unk)) if row[i] != 0)
    val[col] = row[-1] - sum(row[i] * val[i] for i in free if row[i] != 0)
for i, o in enumerate(unk):
    nu[o] = val[i]
print("all orbit values nonnegative:", all(x >= 0 for x in nu), "  min =", float(min(nu)))
print("max |exact - numerical| over orbits:", max(abs(float(nu[o]) - nu_num[o]) for o in range(len(orbits))))
Nex = [[Fr(0)] * 120 for _ in range(120)]
for o, orb in enumerate(orbits):
    for a, b in orb:
        Nex[a][b] = nu[o]; Nex[b][a] = nu[o]
NB = [[sum(Nex[r][s] * B[s][p] for s in range(120) if Nex[r][s] != 0) for p in range(N)] for r in range(120)]
P = [[M[p][q] - sum(B[r][p] * NB[r][q] for r in range(120) if B[r][p] != 0) for q in range(N)] for p in range(N)]
ok = all(all(sum(P[p][q] * k[q] for q in range(N)) == 0 for p in range(N)) for k in ROT + ETA)
print("P k = 0 exactly on the 30 kernel vectors:", ok)
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
print("exact LDL^T: P psd =", psd)
print("float check, smallest nonzero eigenvalue of P:", np.round(np.sort(np.linalg.eigvalsh(RM.to_float(P)))[30], 6))
if psd and ok and all(x >= 0 for x in nu):
    print("\nCERTIFIED EXACTLY:  H + c c^T = P + B^T N B,  P >= 0 (exact LDL^T),  N >= 0 (rational).  Hence m = -1.")
    pickle.dump({'nu': [(x.numerator, x.denominator) for x in nu], 'orbits': orbits, 'zero_orbits': zero_orbits},
                open('exact_certificate.pkl', 'wb'))
    # a human-readable dump of the certificate
    with open('exact_certificate.txt', 'w') as f:
        f.write("# exact copositivity certificate for H + c c^T on the packing cone (m = -1)\n")
        f.write("# orbit  size  representative(a,b)  value (rational)\n")
        for o, orb in enumerate(orbits):
            f.write(f"{o:3d} {len(orb):4d}  {orb[0]}  {nu[o]}\n")
    print("written: exact_certificate.pkl, exact_certificate.txt")

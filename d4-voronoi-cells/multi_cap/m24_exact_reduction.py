#!/usr/bin/env python3
"""
m24_exact_reduction.py

Checks the reduction of the 24-contact case to an inequality between pair
and triple angles (Section "A volume identity exact at the root system").

  (1) exact constants: cap volume, pair volume omega(1/2), the value of
      omega_3(1/2,1/2,1/2) forced by the identity, and tau = E(D4);
  (2) Monte Carlo in R^4 of the cap, pair and triple volumes in B(sqrt 2),
      independent of the closed forms;
  (3) exact combinatorics of D4: positive inner products are 1/2, no four
      roots are pairwise at 1/2, 96 tight pairs, 96 tight triangles,
      every vertex of the 24-cell has norm sqrt 2;
  (4) independent quadrature of omega_3(1/2,1/2,1/2);
  (5) the degree-6 three-point pseudo-configuration: normalisation, the
      linear constraints, positive semidefiniteness of every block on the
      range of the constraint matrices, and its objective with direct
      quadrature of omega_3.  Floating point; not a proof.
"""
import os, sys, itertools
import numpy as np, sympy as sp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from m24_volume_functional import omega2, omega3_one

ok = []
def report(label, cond):
    ok.append(bool(cond)); print("    %-60s %s" % (label, "PASS" if cond else "**FAIL**"))

print("(1) exact constants")
th = sp.symbols('th')
cap = sp.simplify(sp.Rational(16, 3) * sp.pi * sp.integrate(sp.cos(th)**4, (th, sp.pi/4, sp.pi/2)))
I = lambda x: x - sp.tan(x) + (sp.tan(x) + sp.tan(x)**3 / 3) / 4
om = sp.simplify(2 * sp.pi * (I(sp.pi/4) - I(sp.pi/6)))
om3 = sp.simplify((2*sp.pi**2 - 24*cap + 96*om - 8) / 96)
tau = sp.simplify(96*om - 96*om3)
report("cap = pi^2/2 - 4 pi/3", sp.simplify(cap - (sp.pi**2/2 - 4*sp.pi/3)) == 0)
report("omega(1/2) = pi(9 pi + 26 sqrt3 - 72)/54", sp.simplify(om - sp.pi*(9*sp.pi + 26*sp.sqrt(3) - 72)/54) == 0)
report("omega3 = pi^2/16 - pi + 13 sqrt3 pi/27 - 1/12",
       sp.simplify(om3 - (sp.pi**2/16 - sp.pi + 13*sp.sqrt(3)*sp.pi/27 - sp.Rational(1, 12))) == 0)
report("tau = 8 - 32 pi + 10 pi^2", sp.simplify(tau - (8 - 32*sp.pi + 10*sp.pi**2)) == 0)
print("    cap %.12f  omega(1/2) %.12f  omega3 %.12f  tau %.12f" % tuple(float(sp.N(x, 20)) for x in (cap, om, om3, tau)))

print("(2) Monte Carlo in R^4")
rng = np.random.default_rng(2026); n = 4_000_000
X = rng.uniform(-np.sqrt(2), np.sqrt(2), size=(n, 4)); X = X[(X**2).sum(1) <= 2]
box = (2*np.sqrt(2))**4
def mc(ws):
    hit = np.all(X @ np.array(ws).T > 1, axis=1)
    p = hit.sum() / n; return box*p, box*np.sqrt(p*(1-p)/n)
w1 = np.array([1, 0, 0, 0.]); w2 = np.array([.5, np.sqrt(3)/2, 0, 0]); w3 = np.array([.5, 1/(2*np.sqrt(3)), np.sqrt(2/3), 0])
for lab, ws, ex in (("cap", [w1], float(cap)), ("pair at 1/2", [w1, w2], float(om)), ("triple at 1/2", [w1, w2, w3], float(om3))):
    v, e = mc(ws); report("%s: MC %.5f +- %.5f vs %.6f" % (lab, v, e, ex), abs(v - ex) < 4*e + 1e-4)
u = 0.3; w2b = np.array([u, np.sqrt(1-u*u), 0, 0]); v, e = mc([w1, w2b])
report("pair at 0.3: MC %.5f +- %.5f vs %.6f" % (v, e, float(omega2(u))), abs(v - float(omega2(u))) < 4*e + 1e-4)

print("(3) D4 combinatorics, exact")
R = []
for i, j in itertools.combinations(range(4), 2):
    for si in (1, -1):
        for sj in (1, -1):
            r = [0]*4; r[i], r[j] = si, sj; R.append(r)
G = np.array(R) @ np.array(R).T
pos = {int(G[a, b]) for a in range(24) for b in range(24) if a != b and G[a, b] > 0}
report("positive integer inner products between roots are all 1", pos == {1})
four = sum(1 for q in itertools.combinations(range(24), 4) if all(G[a, b] == 1 for a, b in itertools.combinations(q, 2)))
report("no four roots pairwise at inner product 1", four == 0)
report("det of Gram I+J of order 4 is 5, not 4 k^2", sp.Matrix(4, 4, lambda a, b: 2 if a == b else 1).det() == 5)
pairs = sum(1 for a, b in itertools.combinations(range(24), 2) if G[a, b] == 1)
tri = sum(1 for a, b, c in itertools.combinations(range(24), 3) if G[a, b] == G[a, c] == G[b, c] == 1)
report("96 pairs at 1/2 and 96 triangles", pairs == 96 and tri == 96)
verts = [tuple(s*sp.sqrt(2) if k == a else 0 for k in range(4)) for a in range(4) for s in (1, -1)] + \
        [tuple(sp.Rational(e, 1)/sp.sqrt(2) for e in s) for s in itertools.product((1, -1), repeat=4)]
okv = all(all(sum(sp.Rational(r[k])*v_[k] for k in range(4))/sp.sqrt(2) <= 1 for r in R) and
          sp.simplify(sum(x**2 for x in v_)) == 2 for v_ in verts)
report("24 vertices of the 24-cell lie in the cell and have norm sqrt 2", okv and len(verts) == 24)

print("(4) independent quadrature of omega_3(1/2,1/2,1/2)")
q = (4*omega3_one(.5, .5, .5, 360) - omega3_one(.5, .5, .5, 180)) / 3
report("quadrature %.9f vs %.9f" % (q, float(om3)), abs(q - float(om3)) < 2e-6)

print("(5) the degree-6 pseudo-configuration (floating point)")
import three_point_sdp as S
from m24_primal_blocks import blocks
A = np.load(os.path.join(HERE, 'continuation_out', 'm24_primal_d6.npy')); uu, vv, tt, p = A.T
N = 24; T3 = N*(N-1)*(N-2)/6; d = 6; one = np.array([1.0]); LC = [S.leg_coeffs(k) for k in range(d+1)]
B, Gl = blocks(d, uu, vv, tt, LC)
report("p >= 0 and sum p = 1", p.min() >= 0 and abs(p.sum() - 1) < 1e-12)
report("admissible triples (u,v,t <= 1/2, Gram PSD)", np.all(np.c_[uu, vv, tt] <= .5 + 1e-15) and
       np.all(1 + 2*uu*vv*tt - uu**2 - vv**2 - tt**2 >= -1e-12))
lin = min(N + T3*Gl[:, k] @ p for k in range(1, d+1)); report("linear constraints, least value %.3e" % lin, lin > 0)
worst = np.inf
for k in range(d+1):
    n_ = d-k+1; Bk = B[k].reshape(len(uu), n_, n_); Bk = (Bk + Bk.transpose(0, 2, 1))/2*T3
    C0 = N*S.Smat(k, d, one, one, one, LC)[0]
    H = np.einsum('pij,pkj->ik', Bk, Bk)/len(uu) + C0 @ C0.T
    w, V = np.linalg.eigh(H); K = V[:, w > 1e-10*w.max()]; Z = V[:, w <= 1e-10*w.max()]
    if Z.shape[1]:
        resid = max(np.abs(Z.T @ (np.tensordot(p, Bk, 1) + C0)).max(), np.abs(np.einsum('ai,pij->paj', Z.T, Bk)).max())
        report("block %d: common kernel of dimension %d carries nothing (%.1e)" % (k, Z.shape[1], resid), resid < 1e-6)
    M = K.T @ (np.tensordot(p, Bk, 1) + C0) @ K; ev = np.linalg.eigvalsh((M+M.T)/2)
    worst = min(worst, ev.min()/np.abs(ev).max())
report("every block positive definite on its range (least rel. eigenvalue %.2e)" % worst, worst > 0)
idx = np.where(p > 0)[0]
w3v = np.array([0.0 if min(uu[i], vv[i], tt[i]) <= 0 else (4*omega3_one(uu[i], vv[i], tt[i], 180) - omega3_one(uu[i], vv[i], tt[i], 90))/3 for i in idx])
obj = T3*p[idx] @ ((omega2(uu[idx]) + omega2(vv[idx]) + omega2(tt[idx]))/(N-2) - w3v)
# the quadrature above is the n = 180 Richardson value, whose error is of
# order 1e-7 per triple; the margin required here is five orders of
# magnitude wider than that, and the claim being checked is only that the
# degree-6 relaxation admits a measure strictly below tau
report("objective %.6f below tau %.6f by %.4f" % (obj, float(tau), float(tau)-obj),
       obj < float(tau) - 0.01)
print("\nALL PASS" if all(ok) else "\nSOME CHECKS FAILED")

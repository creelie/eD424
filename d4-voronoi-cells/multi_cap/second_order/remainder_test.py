"""
Numerical test of two statements about exact cell volumes of packings near D4
(24 centres y_i = d_i n_i, d_i = 2(1+eta_i), n_i = exp_{u_i}(tau_i), all |y_i-y_j| >= 2):

  (R)  vol - 8 - (2/3) S - (1/2) H(xi,xi)  >= 0      (the third-order remainder is nonnegative)
  (D)  vol - 8 - (2/3) S + (1/2) S^2        >= 0      (the worst-case second-order model is a lower bound)

Writes the tested packings to runs/remainder_rows.dat.

Samples: push patterns eta with a few large entries; tilts on the EDGES of the
first-order cone (maximise a random linear functional subject to the cone),
then scaled down until the true packing holds; S from 0.003 to 0.5.
"""
import sys
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0, '.')
import hessian as Hm
from cone_min import A, H, c_sum

rng = np.random.default_rng(11)
U = Hm.U


def packing_ok(N_, d):
    Y = N_ * d[:, None]
    D = np.linalg.norm(Y[:, None] - Y[None], axis=2) + 10 * np.eye(24)
    return D.min() >= 2 - 1e-12


def cone_edge_tau(eta, g):
    """maximise <g, tau> over the first-order cone at this eta, with |tau_coords| <= 1 (box)"""
    A_tau, A_eta = A[:, :72], A[:, 72:]
    res = linprog(-g, A_ub=A_tau, b_ub=-A_eta @ eta, bounds=[(-1, 1)] * 72, method='highs')
    return res.x if res.success else None


rows = []
minR, minD = np.inf, np.inf
argR = argD = None
for trial in range(600):
    S = 10 ** rng.uniform(-2.5, -0.3)
    kind = rng.integers(0, 4)
    if kind == 0:        # one centre
        eta = np.zeros(24); eta[rng.integers(24)] = S / 2
    elif kind == 1:      # two or three centres
        eta = np.zeros(24); idx = rng.choice(24, rng.integers(2, 4), replace=False); eta[idx] = rng.random(len(idx))
        eta *= S / (2 * eta.sum())
    elif kind == 2:      # spread with a few large
        eta = np.abs(rng.standard_normal(24)) ** 4; eta *= S / (2 * eta.sum())
    else:                # even
        eta = np.full(24, S / 48)
    g = rng.standard_normal(72)
    tau_c = cone_edge_tau(eta, g)
    if tau_c is None:
        continue
    d = 2 * (1 + eta)
    xi = np.zeros(96); xi[72:] = eta
    # scale the tilt down until the true packing holds (the cone is first-order only)
    found = False
    for lam in np.geomspace(1.0, 1e-3, 40):
        xi[:72] = lam * tau_c * S           # tilts of size S along the cone edge
        N_, h = Hm.configuration(xi, 1.0)
        if packing_ok(N_, d):
            found = True; break
    if not found:
        continue
    vol = Hm.volume(N_, h)
    Hq = xi @ H @ xi
    R = vol - 8 - 2 / 3 * S - 0.5 * Hq
    D = vol - 8 - 2 / 3 * S + 0.5 * S ** 2
    tau_norm = np.linalg.norm(xi[:72])
    rows.append((S, kind, tau_norm, Hq, R, D))
    if R < minR: minR, argR = R, (S, kind, tau_norm, Hq)
    if D < minD: minD, argD = D, (S, kind, tau_norm, Hq)
rows = np.array(rows)
print(f"{len(rows)} feasible packings tested, S in [{rows[:,0].min():.4f}, {rows[:,0].max():.4f}], "
      f"||tau|| up to {rows[:,2].max():.4f} (relative to S: up to {(rows[:,2]/rows[:,0]).max():.3f})")
print(f"(R) remainder vol - 8 - (2/3)S - (1/2)H:  min = {minR:+.3e}  at S={argR[0]:.4f}, kind {argR[1]}, ||tau||={argR[2]:.4f}, H={argR[3]:+.4f}")
neg = rows[rows[:, 4] < -1e-12]
print(f"    negative in {len(neg)} cases; their S range: {neg[:,0].min() if len(neg) else float('nan'):.4f} .. {neg[:,0].max() if len(neg) else float('nan'):.4f}")
print(f"(D) vol - 8 - (2/3)S + (1/2)S^2:            min = {minD:+.3e}  at S={argD[0]:.4f}, kind {argD[1]}, ||tau||={argD[2]:.4f}")
negD = rows[rows[:, 5] < -1e-12]
print(f"    negative in {len(negD)} cases")
# size of the remainder relative to S^3, as a measure of the true cubic constant on the cone
ratio = rows[:, 4] / rows[:, 0] ** 3
print(f"remainder / S^3: min {ratio.min():+.3f}, max {ratio.max():+.3f}  (one-centre ray exactly: +1/6 - S/48)")
# how close the tilted configurations get to the one-centre model: H / S^2
print(f"H(xi,xi)/S^2 on the tested packings: min {(rows[:,3]/rows[:,0]**2).min():+.4f}  (cone bound -1)")
# the tested packings, for the figure of the paper (columns: S kind |tau| H R D)
np.savetxt('runs/remainder_rows.dat', rows, header='S kind tau_norm H R D', comments='')

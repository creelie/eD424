"""
Test of a conditional monotonicity: for a feasible configuration (tau, eta),
is  vol(tau, eta) >= vol(0, eta) ?   (un-tilting the directions to the roots
never increases the cell volume, as long as the packing constraints hold).

If true, piece (ii) follows with no expansion at all:
    vol(V(Y)) >= vol(V(Y')) = vol{<x,u_i> <= 1 + delta_i/2} >= vol(Q) = 8,
the last by monotonicity in the support numbers.

(1) second order: min over tau in the cone of  H(tau,eta) - H(0,eta), for many eta;
(2) finite size: exact volumes of random feasible packings vs their un-tilted versions.
"""
import numpy as np, sys
from scipy.optimize import minimize
sys.path.insert(0, '.')
import hessian as Hm
from cone_min import A, H, value   # noqa (A: cone rows, H: Hessian)

rng = np.random.default_rng(3)
U, BASIS = Hm.U, Hm.BASIS

# ---- (1) second order --------------------------------------------------------
print("(1) second order: min_tau [H(tau,eta) - H(0,eta)] on the cone, sum delta = 1")
worst = np.inf
for trial in range(40):
    if trial == 0:
        eta = np.full(24, 1 / 48)
    elif trial == 1:
        eta = np.zeros(24); eta[0] = 0.5
    elif trial == 2:
        eta = np.zeros(24); eta[[0, 1]] = 0.25
    else:
        eta = np.abs(rng.standard_normal(24)) ** 3; eta /= 2 * eta.sum()
    base = np.zeros(96); base[72:] = eta
    v0 = value(base)
    def f(t):
        x = base.copy(); x[:72] = t; return value(x) - v0
    def g(t):
        x = base.copy(); x[:72] = t; return 2 * (H @ x)[:72]
    cons = [{'type': 'ineq', 'fun': lambda t: -(A[:, :72] @ t + A[:, 72:] @ eta)}]
    best = 0.0
    for s in range(6):
        t0 = rng.standard_normal(72) * 0.02 if s else np.zeros(72)
        r = minimize(f, t0, jac=g, constraints=cons, method='SLSQP', options=dict(maxiter=1000, ftol=1e-14))
        if r.success or r.status == 9:
            feas = (A[:, :72] @ r.x + A[:, 72:] @ eta).max()
            if feas < 1e-8:
                best = min(best, r.fun)
    worst = min(worst, best)
    if trial < 4 or best < -1e-8:
        print(f"   eta pattern {trial:2d}: H(0,eta) = {v0:+.5f}, min gain from tilting = {best:+.3e}")
print(f"   over all patterns, the most a tilt can lower the form: {worst:+.3e}  (0 means never)")

# ---- (2) finite size: exact volumes -------------------------------------------
print("\n(2) exact volumes: random feasible packings near D4 vs their un-tilted versions")
def packing_ok(N, d):
    Y = N * d[:, None]
    D = np.linalg.norm(Y[:, None] - Y[None], axis=2) + 10 * np.eye(24)
    return D.min() >= 2 - 1e-12

viol = 0; tested = 0; minratio = np.inf
for trial in range(400):
    S = 10 ** rng.uniform(-2.5, -0.5)          # sum delta from 0.003 to 0.3
    eta = np.abs(rng.standard_normal(24)) ** rng.uniform(1, 4); eta *= S / (2 * eta.sum())
    d = 2 * (1 + eta)
    # random tilt, then shrink it until the packing holds
    xi = np.zeros(96); xi[:72] = rng.standard_normal(72); xi[72:] = eta
    for shrink in np.geomspace(1.0, 1e-4, 60):
        x = xi.copy(); x[:72] *= shrink * S
        N, h = Hm.configuration(x, 1.0)
        if packing_ok(N, d):
            break
    else:
        continue
    tested += 1
    v_tilt = Hm.volume(N, h)
    v_flat = Hm.volume(U, h)
    tau, _ = Hm.unpack(x)
    if v_tilt < v_flat - 1e-9:
        viol += 1
        print(f"   VIOLATION: S={S:.4f} ||tau||={np.linalg.norm(tau):.4f}  vol tilted {v_tilt:.8f} < untilted {v_flat:.8f}")
    minratio = min(minratio, (v_tilt - v_flat))
print(f"   {tested} feasible tilted packings tested; violations of vol(tilted) >= vol(untilted): {viol}")
print(f"   smallest value of vol(tilted) - vol(untilted): {minratio:+.3e}")

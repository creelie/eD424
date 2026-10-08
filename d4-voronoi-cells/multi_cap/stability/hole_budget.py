# Lemma H (hole budget, root-sum-square form of cor:no-room).
# r_1..r_24 = normalised D4 roots, tau = a(sqrt6, rho) = (2 + rho^2)/(2 sqrt6 rho), rho^2 = 4/(1-2 s0), s0 = 1/125.
# Claim: for every unit z, f(z) = sum_i (<z,r_i> - tau)_+^2 >= 6 h^2, h = 1/sqrt2 - tau (equality at the deep holes).
# Proof: the roots form a 5-design, so over the 12 pairs +-r: sum t^2 = 3, sum t^4 = 3/2 (t = |<z,r>|).
# With p(t) = beta t^2 + gamma t^4, gamma = 2 sqrt2 h - 4 h^2, beta = sqrt2 h - gamma: (t - tau)_+^2 >= p(t) on [0,1],
# so f(z) >= 3 beta + 3/2 gamma = 6 h^2.  Part 1 checks the three sign conditions rigorously (rational intervals);
# part 2 checks the design identities and the minimum numerically.
from fractions import Fraction as Fr
from math import isqrt
import itertools, sys

K = 10 ** 40
def sqrt_iv(q):  # rational interval containing sqrt(q), q a positive Fraction
    n, d = q.numerator, q.denominator
    lo = Fr(isqrt(n * d * K * K), d * K)
    hi = lo + Fr(1, d * K)
    assert lo * lo <= q <= hi * hi
    return (lo, hi)
def add(a, b): return (a[0] + b[0], a[1] + b[1])
def neg(a): return (-a[1], -a[0])
def sub(a, b): return add(a, neg(b))
def mul(a, b):
    ps = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]; return (min(ps), max(ps))
def inv(a):
    assert a[0] > 0 or a[1] < 0; return (1 / a[1], 1 / a[0])
def c(x): x = Fr(x); return (x, x)
def fl(a): return '[%.9f, %.9f]' % (float(a[0]), float(a[1]))

s0 = Fr(1, 125)
rho2 = 4 / (1 - 2 * s0)
rho = sqrt_iv(rho2)
s2, s6 = sqrt_iv(Fr(2)), sqrt_iv(Fr(6))
tau = mul(add(c(2), c(rho2)), inv(mul(c(2), mul(s6, rho))))
h = sub(inv(s2), tau)
gamma = sub(mul(c(2), mul(s2, h)), mul(c(4), mul(h, h)))
beta = sub(mul(s2, h), gamma)
cond1 = add(beta, mul(gamma, mul(tau, tau)))               # < 0: p <= 0 on [0, tau]
cond2 = sub(mul(sub(c(1), tau), sub(c(1), tau)), add(beta, gamma))  # > 0: r(1) > 0
lead = neg(gamma)                                           # < 0: r is a downward parabola
bound = add(mul(c(3), beta), mul(c(Fr(3, 2)), gamma))
six_h2 = mul(c(6), mul(h, h))
print('tau    ', fl(tau)); print('h      ', fl(h)); print('gamma  ', fl(gamma)); print('beta   ', fl(beta))
print('beta + gamma tau^2 (need < 0)', fl(cond1)); print('(1-tau)^2 - beta - gamma (need > 0)', fl(cond2))
print('-gamma (need < 0)', fl(lead))
print('3 beta + 3/2 gamma', fl(bound), ' 6 h^2', fl(six_h2))
Fstar = sqrt_iv(six_h2[0])
print('F* = sqrt(6) h >=', '%.9f' % float(Fstar[0]))
ok = cond1[1] < 0 and cond2[0] > 0 and lead[1] < 0
print('RIGOROUS SIGN CONDITIONS:', 'PASS' if ok else 'FAIL')

# r(t) = q(t)/(t - t0)^2 with q(t) = (t - tau)^2 - beta t^2 - gamma t^4 and t0 = 1/sqrt2: q(t0) = q'(t0) = 0 by
# construction (checked in floating point below), r(tau) = -tau^2 (beta + gamma tau^2)/(t0 - tau)^2 > 0 from cond1.
import numpy as np
T, H = float(tau[0]), float(h[0]); G_, B_ = float(gamma[0]), float(beta[0]); t0 = 2 ** -0.5
q = lambda t: np.maximum(t - T, 0) ** 2 - B_ * t ** 2 - G_ * t ** 4
print('q(t0) = %.2e, q\'(t0) = %.2e' % (q(t0), 2 * (t0 - T) - 2 * B_ * t0 - 4 * G_ * t0 ** 3))
tt = np.linspace(0, 1, 2000001); print('min_[0,1] q = %.3e (at %.5f)' % (q(tt).min(), tt[q(tt).argmin()]))
R = []
for i, j in itertools.combinations(range(4), 2):
    for a in (1, -1):
        for b in (1, -1):
            v = np.zeros(4); v[i] = a; v[j] = b; R.append(v / np.sqrt(2))
R = np.array(R)
rng = np.random.default_rng(0)
Z = rng.standard_normal((200000, 4)); Z /= np.linalg.norm(Z, axis=1)[:, None]
P = Z @ R.T
print('design: max |sum t^2 - 6| = %.1e, max |sum t^4 - 3| = %.1e' % (np.abs((P ** 2).sum(1) - 6).max(), np.abs((P ** 4).sum(1) - 3).max()))
f = (np.maximum(P - T, 0) ** 2).sum(1)
print('random z: min f = %.6f vs 6h^2 = %.6f' % (f.min(), 6 * H * H))

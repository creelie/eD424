#!/usr/bin/env python3
"""
gen_hole_budget_lean.py -- writes D4HoleBudget.lean, the exact content of
parts (ii) and (iii) of Corollary "No room beside a near root system".

The rational bounds on sqrt(1/2), tau = a(sqrt6, rho), sqrt6 and sqrt(3/8)
and the 200-row table for part (iii) are computed here with integer square
roots (denominator 10^12); the Lean file then checks every one of them by
kernel computation on Rat, so nothing computed here is trusted.

Usage: python3 gen_hole_budget_lean.py   (from the lean/ directory)
"""
import os
from fractions import Fraction as Fr
from itertools import combinations
from math import isqrt

HERE = os.path.dirname(os.path.abspath(__file__))
S = 10 ** 12
N = 200                       # intervals of e in part (iii)
C = Fr(3059, 10000)           # the threshold of part (iii)
CAP = Fr(515, 10000)          # the table's bound on the right side


def sq_lo(q):
    """Largest m / S with (m / S)^2 <= q."""
    return Fr(isqrt(q.numerator * S * S // q.denominator), S)


def sq_hi(q):
    """Smallest m / S with (m / S)^2 >= q."""
    m = sq_lo(q)
    while m * m < q:
        m += Fr(1, S)
    return m


def lit(x):
    return "(%d / %d : Rat)" % (x.numerator, x.denominator) if x.denominator != 1 else "(%d : Rat)" % x.numerator


rho2 = 4 / (1 - Fr(16, 1000))
tau2 = (2 + rho2) ** 2 / (24 * rho2)
assert rho2 == Fr(500, 123) and tau2 == Fr(139129, 369000)
ulo, uhi = sq_lo(Fr(1, 2)), sq_hi(Fr(1, 2))
tlo, thi = sq_lo(tau2), sq_hi(tau2)
s6lo = sq_lo(Fr(6))
s38lo = sq_lo(Fr(3, 8))
hlo = ulo - thi
assert 6 * hlo * hlo > Fr(5197, 100000)

D2 = 2 * C
emax = D2 / 12
rows, worst = [], Fr(0)
for k in range(N):
    e0, e1 = emax * k / N, emax * (k + 1) / N
    ahi = sq_hi(D2 - 12 * e0)
    L = sq_lo(6 - ahi - e1)
    worst = max(worst, e1 + (ahi + e1) ** 2 / (L + s6lo) ** 2)
    rows.append((ahi, L))
assert worst < CAP

# the 24 roots of D4 as integer vectors +-e_j +-e_k
roots = []
for j, k in combinations(range(4), 2):
    for a in (1, -1):
        for b in (1, -1):
            v = [0, 0, 0, 0]
            v[j], v[k] = a, b
            roots.append(v)
w = ["w0", "w1", "w2", "w3"]


def form(v):
    terms = []
    for i, c in enumerate(v):
        if c:
            terms.append(("+ " if c > 0 else "- ") + w[i])
    s = " ".join(terms)
    return s[2:] if s.startswith("+ ") else "-" + s[2:]


sum2 = " + ".join("(%s)^2" % form(v) for v in roots)
sum4 = " + ".join("(%s)^4" % form(v) for v in roots)

HEAD = """/-
D4HoleBudget.lean: the exact content of parts (ii) and (iii) of the
corollary "No room beside a near root system" (sec:closure of the paper).

Part (ii).  Write u for 1/sqrt2 (so 2u^2 = 1), tau = u - h,
gamma = 2 sqrt2 h - 4h^2 = 4uh - 4h^2, beta = sqrt2 h - gamma = 2uh - gamma, and
q(s) = (s - tau)^2 - beta s^2 - gamma s^4.  Proved over every commutative
ring with `grind` (so verbatim over the reals):

  * q(u) = q'(u) = 0 and q(s) = (s - u)^2 r(s) with r the quadratic below;
  * beta + gamma tau^2 = -2h^2 (1 - 6uh + 2h^2)  and  h^2 r(tau) = -tau^2 (beta + gamma tau^2);
  * (1 - tau)^2 - beta - gamma = (h - u)(h - (3u - 2)), which is q(1);
  * 6 beta + 3 gamma = 12 h^2, that is 3 beta + (3/2) gamma = 6h^2;
  * the design identities: over the 24 integer roots a = +-e_j +-e_k,
    sum <w,a>^2 = 12 |w|^2 and sum <w,a>^4 = 12 |w|^4, so over the normalised
    roots sum t^2 = 6 and sum t^4 = 3 for a unit vector.

By kernel computation on Rat: the roots are closed under negation, have
norm 2, satisfy sum a a^T = 12 I, and at the deep hole e_1 exactly six of
them have <e_1, a> = 1 and the rest at most 0 (so the budget 6h^2 is
attained); rho^2 = 4/(1 - 0.016) = 500/123 and tau^2 = a(sqrt6, rho)^2 =
(2 + rho^2)^2/(24 rho^2) = 139129/369000; with rational bounds on u and tau
checked by squaring, 0.093068 < h < 0.093069, 0.614038 < tau < 0.614039,
6h^2 > 0.05197 and 0.22796 < sqrt6 h < 0.22797.  For every rho in
[2, sqrt6), tau lies in [sqrt6/4, 2/3], so h lies in (0, 0.0948); there
h < u, h < 3u - 2 and 1 - 6uh + 2h^2 > 0, the signs the proof uses.

Part (iii).  With alpha^2 + 12e = D^2 <= 2 * 0.3059, e runs over
[0, D^2/12]; on each of 200 intervals [e0, e1] the table gives rationals
A >= sqrt(D^2 - 12 e0) >= alpha and 0 <= L <= sqrt(6 - A - e1), and the
check is e1 + (A + e1)^2/(L + sqrt6_lo)^2 < 0.0515 < 6h^2.  The bound
e + (alpha + e)^2/(sqrt(6 - alpha - e) + sqrt6)^2 increases in alpha and e,
so it stays below 0.0515.  The matrix identities that lead to it are the
paper's.

Written by gen_hole_budget_lean.py.  No `sorry`, no Mathlib, no native_decide.
-/

namespace D4HoleBudget
open Lean.Grind

section Identities
variable {R : Type} [CommRing R]

def gam (u h : R) : R := 4*u*h - 4*h^2
def bet (u h : R) : R := 2*u*h - gam u h
def q (u h s : R) : R := (s - (u - h))^2 - bet u h * s^2 - gam u h * s^4
def r (u h s : R) : R := -(gam u h)*s^2 - 2*u*(gam u h)*s + 1 - bet u h - 3*(gam u h)*u^2

theorem q_at_contact (u h : R) (hu : 2*u^2 = 1) : q u h u = 0 := by
  unfold q bet gam; grind

theorem q_deriv_at_contact (u h : R) (hu : 2*u^2 = 1) :
    2*(u - (u - h)) - 2*(bet u h)*u - 4*(gam u h)*u^3 = 0 := by
  unfold bet gam; grind

theorem q_factor (u h s : R) (hu : 2*u^2 = 1) : q u h s = (s - u)^2 * r u h s := by
  unfold q r bet gam; grind

theorem below_tau (u h : R) (hu : 2*u^2 = 1) :
    bet u h + gam u h * (u - h)^2 = -2*h^2*(1 - 6*u*h + 2*h^2) := by
  unfold bet gam; grind

theorem r_at_tau (u h : R) (hu : 2*u^2 = 1) :
    h^2 * r u h (u - h) = -(u - h)^2 * (bet u h + gam u h * (u - h)^2) := by
  unfold r bet gam; grind

theorem q_at_one (u h : R) :
    q u h 1 = (1 - (u - h))^2 - bet u h - gam u h := by
  unfold q; grind

theorem at_one (u h : R) (hu : 2*u^2 = 1) :
    (1 - (u - h))^2 - bet u h - gam u h = (h - u)*(h - (3*u - 2)) := by
  unfold bet gam; grind

theorem budget (u h : R) : 6*(bet u h) + 3*(gam u h) = 12*h^2 := by
  unfold bet gam; grind

theorem gam_factor (u h : R) : gam u h = 4*h*(u - h) := by
  unfold gam; grind

"""

DESIGN = """theorem design2 (w0 w1 w2 w3 : R) :
    %s
      = 12*(w0^2 + w1^2 + w2^2 + w3^2) := by
  grind

theorem design4 (w0 w1 w2 w3 : R) :
    %s
      = 12*(w0^2 + w1^2 + w2^2 + w3^2)^2 := by
  grind

end Identities

""" % (sum2, sum4)

FINITE = """def roots : List (List Int) := [
%s]

def idot (a b : List Int) : Int := (List.zipWith (fun x y => x * y) a b).foldl (fun s x => s + x) 0

theorem roots_basic :
    (roots.length == 24 && roots.all (fun a => idot a a == 2) &&
     roots.all (fun a => roots.contains (a.map (fun x => -x)))) = true := by
  decide +kernel

theorem roots_frame :
    ((List.range 4).all fun p => (List.range 4).all fun q =>
      (roots.foldl (fun s a => s + a[p]! * a[q]!) 0) == (if p == q then 12 else 0)) = true := by
  decide +kernel

theorem roots_gram_norm :
    (roots.foldl (fun s a => roots.foldl (fun s b => s + idot a b * idot a b) s) 0) = 576 := by
  decide +kernel

theorem deep_hole :
    ((roots.filter (fun a => idot [1, 0, 0, 0] a == 1)).length == 6 &&
     roots.all (fun a => idot [1, 0, 0, 0] a == 1 || idot [1, 0, 0, 0] a <= 0)) = true := by
  decide +kernel

def rho2 : Rat := 4 / (1 - 16 / 1000)
def tau2 : Rat := (2 + rho2)^2 / (24 * rho2)

theorem rho_tau : rho2 = 500 / 123 /\ tau2 = 139129 / 369000 := by decide +kernel

def ulo : Rat := %s
def uhi : Rat := %s
def tlo : Rat := %s
def thi : Rat := %s
def s6lo : Rat := %s
def s38lo : Rat := %s
def hlo : Rat := ulo - thi
def hhi : Rat := uhi - tlo

theorem sqrt_bounds :
    (0 < ulo && 2 * (ulo * ulo) <= 1 && 1 <= 2 * (uhi * uhi) &&
     0 < tlo && tlo * tlo <= tau2 && tau2 <= thi * thi &&
     0 < s6lo && s6lo * s6lo <= 6 && 0 < s38lo && s38lo * s38lo <= 3 / 8) = true := by
  decide +kernel

theorem h_at_rho :
    (93068 / 1000000 < hlo && hhi < 93069 / 1000000 &&
     614038 / 1000000 < tlo && thi < 614039 / 1000000 &&
     5197 / 100000 < 6 * (hlo * hlo) &&
     (22796 / 100000) * (22796 / 100000) < 6 * (hlo * hlo) &&
     6 * (hhi * hhi) < (22797 / 100000) * (22797 / 100000)) = true := by
  decide +kernel

/-- For rho in [2, sqrt6): tau in [sqrt6/4, 2/3] = [sqrt(3/8), 2/3], so
    h in [ulo - 2/3, uhi - s38lo], where h < u, h < 3u - 2 and 1 - 6uh > 0. -/
theorem h_signs_all_rho :
    (0 < ulo - 2 / 3 && uhi - s38lo < 948 / 10000 &&
     uhi - s38lo < ulo && uhi - s38lo < 3 * ulo - 2 &&
     0 < 1 - 6 * uhi * (uhi - s38lo)) = true := by
  decide +kernel

def cThr : Rat := 3059 / 10000
def D2 : Rat := 2 * cThr
def emax : Rat := D2 / 12
def nInt : Nat := %d

def rows : List (Prod Rat Rat) := [
%s]

def rowOk (k : Nat) (row : Prod Rat Rat) : Bool :=
  let e0 := emax * (k : Rat) / (nInt : Rat)
  let e1 := emax * ((k : Rat) + 1) / (nInt : Rat)
  let A := row.1
  let L := row.2
  0 <= A && D2 - 12 * e0 <= A * A &&
  0 <= L && 0 <= 6 - A - e1 && L * L <= 6 - A - e1 &&
  e1 + (A + e1) * (A + e1) / ((L + s6lo) * (L + s6lo)) < 515 / 10000

theorem table_iii :
    (rows.length == nInt && ((List.range nInt).zip rows).all (fun p => rowOk p.1 p.2)) = true := by
  decide +kernel

theorem cap_below_budget : (515 / 10000 : Rat) < 6 * (hlo * hlo) := by decide +kernel

end D4HoleBudget

#print axioms D4HoleBudget.q_factor
#print axioms D4HoleBudget.design4
#print axioms D4HoleBudget.roots_frame
#print axioms D4HoleBudget.h_at_rho
#print axioms D4HoleBudget.h_signs_all_rho
#print axioms D4HoleBudget.table_iii
#print axioms D4HoleBudget.cap_below_budget
""" % (
    ",\n".join("  [%s]" % ", ".join(str(x) for x in v) for v in roots),
    lit(ulo), lit(uhi), lit(tlo), lit(thi), lit(s6lo), lit(s38lo),
    N,
    ",\n".join("  (%s, %s)" % (lit(a), lit(b)) for a, b in rows),
)

out = HEAD + DESIGN + FINITE
assert all(ord(ch) < 128 for ch in out)
open(os.path.join(HERE, "D4HoleBudget.lean"), "w").write(out)
print("wrote D4HoleBudget.lean: %d intervals, table maximum %.6f < %s, 6h^2 >= %.8f"
      % (N, float(worst), float(CAP), float(6 * hlo * hlo)))

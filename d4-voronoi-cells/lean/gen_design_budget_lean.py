#!/usr/bin/env python3
"""
gen_design_budget_lean.py: writes D4DesignBudget.lean, the Lean check of the
exact content of the proposition "No room from the design defects" (Section
21.1 of the paper).

The polynomial p = sum_{k=0}^{5} p_k G_k, with G_k = U_k/(k+1) the Gegenbauer
polynomials of S^3 (U_k the Chebyshev polynomials of the second kind), and the
weights c_k are those of the paper.  The Lean file checks, by kernel
computation on Rat:

  * the monomial coefficients of p;
  * the two-square identity of the paper's proof: with the quadratics A and B
    below, R = -p - (1 + t) A^2 - (th - t) B^2 has constant term larger than
    the sum of the absolute values of its other coefficients, so R > 0 on
    [-1, 1] and p < 0 on [-1, th];
  * independently, p(t) < 0 on [-1, th], where th = 0.614039 > tau
    (tau^2 = 139129/369000), by a Taylor bound about the midpoint of each
    subinterval listed in the file, which cover the range without gaps;
  * (24 p_0)^2 > 0.26785 * sum_k p_k^2 / c_k;
  * the root system has S_1 = ... = S_5 = 0.

The addition formula and the Cauchy-Schwarz step are the paper's.  The
subintervals are found here by bisection with the same bound that the Lean
file computes, so every one of them passes there as well.

Run: python gen_design_budget_lean.py   (writes D4DesignBudget.lean beside this script)
"""
import itertools
import os
from fractions import Fraction as Fr

P_GEGEN = [Fr(39954677, 453230000000), Fr(41, 89653), Fr(29, 33806),
           Fr(17, 14206), Fr(95, 98183), Fr(52, 86257)]
C_W = [Fr(103, 963), Fr(191, 963), Fr(280, 963), Fr(230, 963), Fr(159, 963)]
TH = Fr(614039, 10 ** 6)
PHI = Fr(26785, 10 ** 5)
A_SQ = [Fr(26327, 10 ** 8), Fr(59724, 10 ** 8), Fr(112736, 10 ** 8)]
B_SQ = [Fr(260109, 10 ** 8), Fr(4473253, 10 ** 8), Fr(5671392, 10 ** 8)]


def cheb_u(k):
    a, b = [1], [0, 2]
    if k == 0:
        return a
    for _ in range(k - 1):
        c = [0] + [2 * x for x in b]
        for i, x in enumerate(a):
            c[i] -= x
        a, b = b, c
    return b


US = [cheb_u(k) for k in range(6)]


def monomial(pg):
    out = [Fr(0)] * len(pg)
    for k, pk in enumerate(pg):
        for i, x in enumerate(US[k]):
            out[i] += pk * x / (k + 1)
    return out


def shift(cs, m):
    """coefficients of cs(m + d) in d (Horner-style synthetic division)"""
    c = list(cs)
    n = len(c)
    for j in range(n - 1):
        for i in range(n - 2, j - 1, -1):
            c[i] += m * c[i + 1]
    return c


def upper(cs, a, b):
    """an upper bound for cs on [a, b]: c_0 + sum_i |c_i| r^i about the midpoint"""
    m, r = (a + b) / 2, (b - a) / 2
    c = shift(cs, m)
    return c[0] + sum(abs(x) * r ** i for i, x in enumerate(c) if i > 0)


def cover(cs, lo, hi, out, depth=0):
    if upper(cs, lo, hi) < 0:
        out.append((lo, hi))
        return
    assert depth < 60, "no cover"
    mid = (lo + hi) / 2
    cover(cs, lo, mid, out, depth + 1)
    cover(cs, mid, hi, out, depth + 1)


def pmul(a, b):
    out = [Fr(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def padd(a, b):
    n = max(len(a), len(b))
    return [(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(n)]


def two_square_rest(pm):
    """R = -p - (1 + t) A^2 - (th - t) B^2, constant term first"""
    sq = padd(pmul([1, 1], pmul(A_SQ, A_SQ)), pmul([TH, -1], pmul(B_SQ, B_SQ)))
    return padd([-x for x in pm], [-x for x in sq])


def lean_rat(x):
    x = Fr(x)
    return f"({x.numerator} : Rat)" if x.denominator == 1 else f"({x.numerator} / {x.denominator} : Rat)"


def lst(xs):
    return "[" + ", ".join(lean_rat(x) for x in xs) + "]"


def main():
    pm = monomial(P_GEGEN)
    rest = two_square_rest(pm)
    assert rest[0] - sum(abs(x) for x in rest[1:]) > Fr(48, 10 ** 9)
    I1 = []
    cover(pm, Fr(-1), TH, I1)
    phi_ok = (24 * P_GEGEN[0]) ** 2 > PHI * sum(x * x / c for x, c in zip(P_GEGEN[1:], C_W))
    assert phi_ok
    cheb_lean = "[" + ", ".join("[" + ", ".join(str(x) for x in u) + "]" for u in US) + "]"
    L = []
    L.append(f"""/-
D4DesignBudget.lean: the exact content of the proposition "No room from the
design defects" (sec:closure of the paper).

With G_k = U_k/(k+1) (U_k the Chebyshev polynomials of the second kind, so
G_k(1) = 1) and p = sum_(k=0..5) p_k G_k for the rationals p_k below:

  * pmono is the list of monomial coefficients of p (checked from the U_k);
  * p(t) < 0 on [-1, th], with th = 0.614039 > tau, where
    tau^2 = 139129/369000 (as in D4HoleBudget), proved in the paper by the
    identity -p = (1 + t) A^2 + (th - t) B^2 + R: theorem two_square checks
    that R has constant term larger than the sum of the absolute values of
    its other coefficients (so R > 0 on [-1, 1]);
  * independently, the same sign by a Taylor bound about the midpoint of each
    of {len(I1)} subintervals that cover the range without gaps;
  * (24 p_0)^2 > 0.26785 * sum_(k=1..5) p_k^2 / c_k for the weights c_k;
  * over the 24 roots +-e_j +-e_k, sum over ordered pairs of G_k(<a,b>/2)
    is 0 for k = 1, ..., 5 (the root system is a spherical 5-design).

The addition formula and the Cauchy-Schwarz step of the proof are the
paper's.

Written by gen_design_budget_lean.py.  No `sorry`, no Mathlib, no native_decide.
-/

namespace D4DesignBudget

def cheb : List (List Int) := {cheb_lean}

def pgegen : List Rat := {lst(P_GEGEN)}

def weights : List Rat := {lst(C_W)}

def pmono : List Rat := {lst(pm)}

def th : Rat := {lean_rat(TH)}

def sqA : List Rat := {lst(A_SQ)}

def sqB : List Rat := {lst(B_SQ)}
""")
    L.append("""/-- coefficient i of sum_k g_k U_k/(k+1) -/
def monoOf (g : List Rat) (i : Nat) : Rat :=
  (List.range g.length).foldl (fun s k =>
    s + g.getD k 0 * ((cheb.getD k []).getD i 0 : Int) / ((k + 1 : Nat) : Rat)) 0

theorem pmono_ok :
    (pmono.length == 6 && (List.range 6).all (fun i => monoOf pgegen i == pmono.getD i 0)) = true := by
  decide +kernel

/-- one pass of synthetic division by (t - m) from the top down to index j -/
def shiftPass (m : Rat) (j : Nat) (c : List Rat) : List Rat :=
  (List.range (c.length - 1 - j)).foldl (fun c t =>
    let i := c.length - 2 - t
    c.set i (c.getD i 0 + m * c.getD (i + 1) 0)) c

/-- the coefficients of cs(m + d) as a polynomial in d -/
def shift (cs : List Rat) (m : Rat) : List Rat :=
  (List.range (cs.length - 1)).foldl (fun c j => shiftPass m j c) cs

def rabs (x : Rat) : Rat := if x < 0 then -x else x

/-- the product of two polynomials, constant term first -/
def pmul (a b : List Rat) : List Rat :=
  (List.range (a.length + b.length - 1)).map (fun n =>
    (List.range (n + 1)).foldl (fun s i => s + a.getD i 0 * b.getD (n - i) 0) 0)

/-- the sum of two polynomials, constant term first -/
def padd (a b : List Rat) : List Rat :=
  (List.range (max a.length b.length)).map (fun i => a.getD i 0 + b.getD i 0)

/-- R = -p - (1 + t) A^2 - (th - t) B^2 -/
def rest : List Rat :=
  padd (pmono.map (fun x => -x))
    ((padd (pmul [1, 1] (pmul sqA sqA)) (pmul [th, -1] (pmul sqB sqB))).map (fun x => -x))

/-- R > 0 on [-1, 1], because its constant term beats the other coefficients;
so -p = (1 + t) A^2 + (th - t) B^2 + R > 0 on [-1, th] -/
theorem two_square :
    (rest.length == 6 &&
      decide (rest.tail.foldl (fun s x => s + rabs x) 0 < rest.getD 0 0)) = true := by
  decide +kernel

/-- an upper bound for the polynomial cs on [a, b]: c_0 + sum |c_i| r^i about the midpoint -/
def upper (cs : List Rat) (a b : Rat) : Rat :=
  let m := (a + b) / 2
  let r := (b - a) / 2
  let c := shift cs m
  (c.tail.foldl (fun (acc : Rat × Rat) x => (acc.1 + rabs x * (acc.2 * r), acc.2 * r)) (c.getD 0 0, 1)).1

/-- the intervals cover [lo, hi] from left to right without gaps -/
def covers : List (Rat × Rat) -> Rat -> Rat -> Bool
  | [], lo, hi => lo == hi
  | (a, b) :: rest, lo, hi => a == lo && a < b && covers rest b hi
""")
    L.append("def cover1 : List (Rat × Rat) := [\n  " +
             ",\n  ".join(f"({lean_rat(a)}, {lean_rat(b)})" for a, b in I1) + "]\n")
    L.append("""theorem p_negative_below_th :
    (covers cover1 (-1) th && cover1.all (fun ab => upper pmono ab.1 ab.2 < 0)) = true := by
  decide +kernel

theorem th_above_tau : (0 < th && 139129 / 369000 < th * th) = true := by
  decide +kernel

theorem threshold :
    (26785 / 100000 * ((List.range 5).foldl (fun s k =>
        s + pgegen.getD (k + 1) 0 * pgegen.getD (k + 1) 0 / weights.getD k 1) 0) <
      (24 * pgegen.getD 0 0) * (24 * pgegen.getD 0 0)) = true := by
  decide +kernel

def roots : List (List Int) := [""" + ",\n  ".join(
        "[" + ", ".join(str(x) for x in v) + "]"
        for v in [tuple((s1 if t == i else s2 if t == j else 0) for t in range(4))
                  for i, j in itertools.combinations(range(4), 2) for s1 in (1, -1) for s2 in (1, -1)]) + """]

def idot (a b : List Int) : Int := (List.zipWith (fun x y => x * y) a b).foldl (fun s x => s + x) 0

/-- G_k(x) = U_k(x)/(k+1) -/
def gk (k : Nat) (x : Rat) : Rat :=
  (((cheb.getD k []).map (fun (c : Int) => (c : Rat))).foldr (fun cf s => cf + x * s) 0) / ((k + 1 : Nat) : Rat)

theorem roots_design :
    (roots.length == 24 && (List.range 5).all (fun k =>
      roots.foldl (fun s a => roots.foldl (fun s b => s + gk (k + 1) ((idot a b : Rat) / 2)) s) 0 == 0)) = true := by
  decide +kernel

end D4DesignBudget

#print axioms D4DesignBudget.two_square
#print axioms D4DesignBudget.p_negative_below_th
#print axioms D4DesignBudget.threshold
#print axioms D4DesignBudget.roots_design
""")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "D4DesignBudget.lean"), "w") as f:
        f.write("\n".join(L))
    print(f"wrote D4DesignBudget.lean ({len(I1)} intervals)")


if __name__ == "__main__":
    main()

/-
CountDomain.lean: the check of a count certificate (thm:count31) inside Lean.

A certificate (CountData31.lean) consists of rationals c1, c2, dmax, m, a vector z
and symmetric matrices A_0, ..., A_D of order r + 1.  With p(d) the vector of
Chebyshev polynomials T_0, ..., T_r at x = (2d - c1)/c2 and G_k = U_k/(k + 1),

  K(d, d', u) = sum_k G_k(u) p(d)^T A_k p(d').

This module establishes what multi_cap/radial_count_check.py establishes:

  1. A_1, ..., A_D are positive semidefinite and A_0 is positive definite, by an
     exact LDL^T over the rationals; t = z^T A_0^{-1} z rounded up to a multiple of
     2^-48, and [[A_0, z], [z^T, t]] is positive semidefinite by the same test;
  2. K(d, d', u) <= Pi(d/2, d'/2, u) for 2 <= d <= d' <= dmax and
     -1 <= u <= a(d, d') = (d^2 + d'^2 - 4)/(2 d d'), where dmax^2 > 6 and Pi = 0
     beyond sqrt 6;
  3. S(d) + K(d, d, 1)/2 - z . p(d) <= m on [2, dmax];
  4. m < 0 and M m + t/2 < 9 pi^2/8 - 8.

For 2 the tensor Bernstein coefficients of K on the root box
[2, dmax]^2 x [-1, a(dmax, dmax)] are computed in exact rational arithmetic from
the certificate and rounded up to multiples of 2^-320.  Rounding a coefficient up
can only raise the polynomial it represents, since the Bernstein basis is
nonnegative, and the subdivision at midpoints (de Casteljau) takes averages, so it
is carried out exactly in dyadic arithmetic and the largest coefficient on a box
bounds K there from above.  Pi is nonincreasing in d and d' and nondecreasing in u
(lem:pair-closed), so on a box it is at least its value at the corner
(d_hi, d'_hi, u_lo), which is bounded below from its closed form in outward-rounded
dyadic interval arithmetic with 256 fractional bits: square roots from Nat.sqrt,
arctan from its series after the usual reductions, pi from Machin's formula, and
sin and cos from their Taylor series with the remainder added on both sides.  A
box is closed when the bound for K is at most 0 or at most that lower bound for
Pi, set aside when it lies in d > d' or wholly beyond u = a(d, d') (exact rational
tests), and halved along its widest side otherwise, the u side weighted as in the
Python check.  For 3 the same is done in one variable, S being decreasing and
bounded above at the left end of each interval.

The interval routines are those of lean/certificate/D4Omega.lean, repeated here
so that this project depends on nothing else.
-/
import Std.Data.HashMap

/-! ## Dyadic numbers -/

structure Dy where
  n : Int
  e : Nat
deriving Inhabited, Repr

namespace Dy
def shl (x : Int) (k : Nat) : Int := x * Int.ofNat (Nat.shiftLeft 1 k)
def align (a b : Dy) : Int × Int × Nat :=
  if a.e ≥ b.e then (a.n, shl b.n (a.e - b.e), a.e) else (shl a.n (b.e - a.e), b.n, b.e)
def add (a b : Dy) : Dy := let (x, y, e) := align a b; ⟨x + y, e⟩
def sub (a b : Dy) : Dy := let (x, y, e) := align a b; ⟨x - y, e⟩
def mul (a b : Dy) : Dy := ⟨a.n * b.n, a.e + b.e⟩
def neg (a : Dy) : Dy := ⟨-a.n, a.e⟩
def le (a b : Dy) : Bool := let (x, y, _) := align a b; x ≤ y
def lt (a b : Dy) : Bool := let (x, y, _) := align a b; x < y
def max (a b : Dy) : Dy := if le a b then b else a
def min (a b : Dy) : Dy := if le a b then a else b
def abs (a : Dy) : Dy := ⟨a.n.natAbs, a.e⟩
def half (a : Dy) : Dy := ⟨a.n, a.e + 1⟩
def zero : Dy := ⟨0, 0⟩
def one : Dy := ⟨1, 0⟩
def toRat (a : Dy) : Rat := (a.n : Rat) / ((2 ^ a.e : Nat) : Rat)
/-- The least multiple of 2^-e at or above q. -/
def ceilRat (q : Rat) (e : Nat) : Dy := ⟨(q * ((2 ^ e : Nat) : Rat)).ceil, e⟩
end Dy

/-! ## Outward-rounded intervals with 256 fractional bits -/

namespace Rd
def P : Nat := 256
def pow2 (k : Nat) : Int := Int.ofNat (Nat.shiftLeft 1 k)
def down (a : Dy) : Dy := if a.e ≤ P then a else ⟨Int.fdiv a.n (pow2 (a.e - P)), P⟩
def up (a : Dy) : Dy := if a.e ≤ P then a else ⟨-(Int.fdiv (-a.n) (pow2 (a.e - P))), P⟩
end Rd

structure RI where
  lo : Dy
  hi : Dy
deriving Inhabited

namespace RI
open Rd
def pt (a : Dy) : RI := ⟨a, a⟩
def ofRat (q : Rat) : RI :=
  ⟨⟨(q * ((2 ^ P : Nat) : Rat)).floor, P⟩, ⟨(q * ((2 ^ P : Nat) : Rat)).ceil, P⟩⟩
def ofNat (n : Nat) : RI := pt ⟨n, 0⟩
def huge : RI := ⟨⟨-(pow2 400), 0⟩, ⟨pow2 400, 0⟩⟩
def add (x y : RI) : RI := ⟨down (x.lo.add y.lo), up (x.hi.add y.hi)⟩
def neg (x : RI) : RI := ⟨x.hi.neg, x.lo.neg⟩
def sub (x y : RI) : RI := add x (neg y)
def mul (x y : RI) : RI :=
  let p1 := x.lo.mul y.lo; let p2 := x.lo.mul y.hi; let p3 := x.hi.mul y.lo; let p4 := x.hi.mul y.hi
  ⟨down (Dy.min (Dy.min p1 p2) (Dy.min p3 p4)), up (Dy.max (Dy.max p1 p2) (Dy.max p3 p4))⟩
def scaleRat (q : Rat) (x : RI) : RI := mul (ofRat q) x
def divDown (a b : Dy) : Dy := ⟨Int.fdiv (a.n * pow2 (b.e + P)) (b.n * pow2 a.e), P⟩
def divUp (a b : Dy) : Dy := ⟨-(Int.fdiv (-(a.n * pow2 (b.e + P))) (b.n * pow2 a.e)), P⟩
def isPos (x : RI) : Bool := x.lo.n > 0
/-- x / y for y > 0; the huge interval otherwise, which no check passes. -/
def div (x y : RI) : RI :=
  if !isPos y then huge else
  let c := [divDown x.lo y.lo, divDown x.lo y.hi, divDown x.hi y.lo, divDown x.hi y.hi]
  let d := [divUp x.lo y.lo, divUp x.lo y.hi, divUp x.hi y.lo, divUp x.hi y.hi]
  ⟨c.foldl Dy.min (c.getD 0 Dy.zero), d.foldl Dy.max (d.getD 0 Dy.zero)⟩
def sqrtDown (a : Dy) : Dy :=
  if a.n ≤ 0 then Dy.zero else
  let m : Nat := (if a.e ≤ 2 * P then a.n * pow2 (2 * P - a.e) else Int.fdiv a.n (pow2 (a.e - 2 * P))).toNat
  ⟨Nat.sqrt m, P⟩
def sqrtUp (a : Dy) : Dy :=
  if a.n ≤ 0 then Dy.zero else
  let m : Nat := (if a.e ≤ 2 * P then a.n * pow2 (2 * P - a.e) else -(Int.fdiv (-a.n) (pow2 (a.e - 2 * P)))).toNat
  let r := Nat.sqrt m
  ⟨if r * r ≥ m then r else r + 1, P⟩
/-- The square root of an interval whose lower end is at least 0 (the lower end is
clamped at 0, which is valid when the true argument is nonnegative). -/
def sqrt (x : RI) : RI := ⟨sqrtDown x.lo, sqrtUp x.hi⟩
def absHi (x : RI) : Dy := Dy.max x.lo.abs x.hi.abs
def pow (x : RI) : Nat → RI
  | 0 => pt Dy.one
  | n + 1 => mul (pow x n) x
def hull (x y : RI) : RI := ⟨Dy.min x.lo y.lo, Dy.max x.hi y.hi⟩
/-- The interval of max(a, b) for a in x and b in y. -/
def max (x y : RI) : RI := ⟨Dy.max x.lo y.lo, Dy.max x.hi y.hi⟩
/-- The interval of min(a, b) for a in x and b in y. -/
def min (x y : RI) : RI := ⟨Dy.min x.lo y.lo, Dy.min x.hi y.hi⟩
end RI

/-! ## arctan, pi, sin, cos and tan -/

/-- arctan on an interval x with |x| <= 1/2: 120 terms of the series, with the
remainder |x|^241/241 added on both sides. -/
def atanSeries (x : RI) : RI := Id.run do
  let x2 := RI.mul x x
  let mut term := x
  let mut acc := RI.pt Dy.zero
  for k in [0:120] do
    let t := RI.div term (RI.ofNat (2 * k + 1))
    acc := if k % 2 == 0 then RI.add acc t else RI.sub acc t
    term := RI.mul term x2
  let b := RI.absHi x
  let rem := RI.div (RI.pow (RI.pt b) 241) (RI.ofNat 241)
  return ⟨Rd.down (acc.lo.sub rem.hi), Rd.up (acc.hi.add rem.hi)⟩

/-- pi by Machin's formula. -/
def piI : RI :=
  RI.sub (RI.scaleRat 16 (atanSeries (RI.ofRat (1/5)))) (RI.scaleRat 4 (atanSeries (RI.ofRat (1/239))))

def halfPiI : RI := RI.scaleRat (1/2) piI

/-- arctan at a point a >= 0: pi/2 - arctan(1/a) above 2, the half-angle reduction
2 arctan(a/(1 + sqrt(1 + a^2))) on (1/2, 2], the series below. -/
def atanNonneg : Nat → Dy → RI
  | 0, _ => RI.huge
  | fuel + 1, a =>
    if Dy.lt ⟨2, 0⟩ a then
      let inv := RI.div (RI.pt Dy.one) (RI.pt a)
      let t := RI.hull (atanNonneg fuel inv.lo) (atanNonneg fuel inv.hi)
      RI.sub halfPiI t
    else if Dy.lt ⟨1, 1⟩ a then
      let b := RI.div (RI.pt a) (RI.add (RI.pt Dy.one) (RI.sqrt (RI.add (RI.pt Dy.one) (RI.mul (RI.pt a) (RI.pt a)))))
      let t := RI.hull (atanNonneg fuel b.lo) (atanNonneg fuel b.hi)
      RI.add t t
    else atanSeries (RI.pt a)

def atanPt (a : Dy) : RI := if a.n < 0 then RI.neg (atanNonneg 4 a.neg) else atanNonneg 4 a

/-- arctan on an interval, by monotonicity. -/
def atanI (x : RI) : RI := ⟨(atanPt x.lo).lo, (atanPt x.hi).hi⟩

/-- sin and cos at a point a with |a| <= 1: 60 terms of each series, with the
remainders |a|^121/121! and |a|^120/120! added on both sides; the huge interval
for |a| > 1. -/
def sinCosPt (a : Dy) : RI × RI := Id.run do
  if Dy.lt Dy.one a.abs then return (RI.huge, RI.huge)
  let x := RI.pt a
  let x2 := RI.mul x x
  let mut s := RI.pt Dy.zero
  let mut c := RI.pt Dy.zero
  let mut ts := x                -- a^(2k+1)/(2k+1)!
  let mut tc := RI.pt Dy.one     -- a^(2k)/(2k)!
  for k in [0:60] do
    s := if k % 2 == 0 then RI.add s ts else RI.sub s ts
    c := if k % 2 == 0 then RI.add c tc else RI.sub c tc
    ts := RI.div (RI.mul ts x2) (RI.ofNat ((2 * k + 2) * (2 * k + 3)))
    tc := RI.div (RI.mul tc x2) (RI.ofNat ((2 * k + 1) * (2 * k + 2)))
  -- after the loop ts = |a|^121/121! and tc = |a|^120/120! in absolute value
  let rs := RI.absHi ts
  let rc := RI.absHi tc
  return (⟨Rd.down (s.lo.sub rs), Rd.up (s.hi.add rs)⟩, ⟨Rd.down (c.lo.sub rc), Rd.up (c.hi.add rc)⟩)

/-- tan at a point a with |a| <= 1, where cos a > 1/2. -/
def tanPt (a : Dy) : RI := let (s, c) := sinCosPt a; RI.div s c

/-- tan on an interval inside [-1, 1], by monotonicity. -/
def tanI (x : RI) : RI := ⟨(tanPt x.lo).lo, (tanPt x.hi).hi⟩

/-! ## The cap S and the pair term Pi -/

def r2 : Rat := 3 / 2

/-- S(d) for 2 <= d, from h = d/2: the volume of the cap of B(sqrt(3/2)) beyond the
hyperplane at distance h,
(4 pi/3) [(27/32)(pi/2) - (h/8)(5R^2 - 2h^2)(R^2 - h^2)^(1/2) - (27/32) arctan(h/(R^2 - h^2)^(1/2))],
and 0 when d^2 >= 6. -/
def capS (d : Rat) : RI :=
  if d * d ≥ 6 then RI.pt Dy.zero else
  let h := RI.ofRat (d / 2)
  let sq := RI.sqrt (RI.ofRat (r2 - d * d / 4))
  let asin := atanI (RI.div h sq)
  let inner := RI.add (RI.mul (RI.scaleRat (1/8) h) (RI.mul (RI.ofRat (5 * r2 - d * d / 2)) sq))
                      (RI.scaleRat (27/32) asin)
  RI.mul (RI.scaleRat (4/3) piI) (RI.sub (RI.scaleRat (27/64) piI) inner)

/-- G(h, x) = R^4 x - 2 R^2 h^2 tan x + h^4 (tan x + tan^3 x / 3), with R^2 = 3/2. -/
def gI (h : Rat) (x : RI) : RI :=
  let tx := tanI x
  let h2 := h * h
  RI.add (RI.sub (RI.scaleRat (9/4) x) (RI.scaleRat (3 * h2) tx))
         (RI.scaleRat (h2 * h2) (RI.add tx (RI.scaleRat (1/3) (RI.pow tx 3))))

/-- A lower bound for Pi(d1/2, d2/2, u) (lem:pair-closed), at least 0:
gamma = arccos u, alpha_k = arccos(h_k/R), L = max(-alpha_1, gamma - alpha_2),
U = min(alpha_1, gamma + alpha_2), phi_c = atan2(h_2 - h_1 u, h_1 sin gamma) clipped
to [L, U], and Pi = (pi/4)(G(h_2, phi_c - gamma) - G(h_2, L - gamma) + G(h_1, U) - G(h_1, phi_c))
when L < U.  Pi >= 0 always, so 0 is returned whenever the intervals do not decide
L < U, and also outside the open domain. -/
def piLower (d1 d2 u : Rat) : Dy :=
  if d1 * d1 ≥ 6 || d2 * d2 ≥ 6 || u ≤ -1 || u ≥ 1 then Dy.zero else
  let h1 := d1 / 2
  let h2 := d2 / 2
  let su := RI.sqrt (RI.ofRat (1 - u * u))
  let g := RI.sub halfPiI (atanI (RI.div (RI.ofRat u) su))
  let a1 := atanI (RI.div (RI.sqrt (RI.ofRat (r2 - h1 * h1))) (RI.ofRat h1))
  let a2 := atanI (RI.div (RI.sqrt (RI.ofRat (r2 - h2 * h2))) (RI.ofRat h2))
  let L := RI.max (RI.neg a1) (RI.sub g a2)
  let U := RI.min a1 (RI.add g a2)
  if !(Dy.lt L.hi U.lo) then Dy.zero else
  let pc0 := atanI (RI.div (RI.ofRat (h2 - h1 * u)) (RI.scaleRat h1 su))
  let pc := RI.min (RI.max pc0 L) U
  let v := RI.add (RI.sub (gI h2 (RI.sub pc g)) (gI h2 (RI.sub L g)))
                  (RI.sub (gI h1 U) (gI h1 pc))
  let val := RI.mul (RI.scaleRat (1/4) piI) v
  Dy.max val.lo Dy.zero

/-! ## Exact linear algebra -/

/-- Exact LDL^T: some pivots if the matrix is positive semidefinite (a zero pivot
must have a zero column below it), none otherwise. -/
def ldlPivots (M0 : Array (Array Rat)) : Option (Array Rat) := Id.run do
  let n := M0.size
  let mut A := M0
  let mut piv : Array Rat := #[]
  for k in [0:n] do
    let p := A[k]![k]!
    if p < 0 then return none
    if p == 0 then
      for i in [k+1:n] do
        if A[i]![k]! != 0 then return none
      piv := piv.push 0
    else
      piv := piv.push p
      for i in [k+1:n] do
        let f := A[i]![k]! / p
        if f != 0 then
          let rowk := A[k]!
          let mut rowi := A[i]!
          for j in [k+1:n] do
            rowi := rowi.set! j (rowi[j]! - f * rowk[j]!)
          A := A.set! i rowi
  return some piv

def isPSD (M0 : Array (Array Rat)) : Bool := (ldlPivots M0).isSome
def isPD (M0 : Array (Array Rat)) : Bool :=
  match ldlPivots M0 with
  | some p => p.all (fun x => x > 0)
  | none => false

/-- Solve M x = b exactly (Gauss-Jordan with the first nonzero pivot). -/
def solveExact (M0 : Array (Array Rat)) (b : Array Rat) : Array Rat := Id.run do
  let n := M0.size
  let mut A : Array (Array Rat) := (Array.range n).map fun i => (M0[i]!).push b[i]!
  for k in [0:n] do
    let mut p := k
    for i in [k:n] do
      if A[p]![k]! == 0 && A[i]![k]! != 0 then p := i
    let tmp := A[k]!
    A := (A.set! k A[p]!).set! p tmp
    let rowk := A[k]!
    for i in [0:n] do
      if i != k && A[i]![k]! != 0 then
        let f := A[i]![k]! / rowk[k]!
        A := A.set! i ((Array.range (n + 1)).map fun j => A[i]![j]! - f * rowk[j]!)
  return (Array.range n).map fun i => A[i]![n]! / A[i]![i]!

def isSymmetric (M0 : Array (Array Rat)) : Bool :=
  (List.range M0.size).all fun i => (List.range M0.size).all fun j => M0[i]![j]! == M0[j]![i]!

/-! ## Univariate polynomials in s on [0, 1] and their Bernstein coefficients -/

abbrev UPoly := Array Rat

def UPoly.mul (p q : UPoly) : UPoly := Id.run do
  let mut out : Array Rat := Array.replicate (p.size + q.size - 1) 0
  for i in [0:p.size] do
    if p[i]! != 0 then
      for j in [0:q.size] do
        out := out.set! (i + j) (out[i + j]! + p[i]! * q[j]!)
  return out

def UPoly.lin (p q : UPoly) (cp cq : Rat) : UPoly :=
  let n := Nat.max p.size q.size
  (Array.range n).map fun i => cp * p.getD i 0 + cq * q.getD i 0

/-- T_0, ..., T_n of x = alpha + beta s, as polynomials in s. -/
def chebT (alpha beta : Rat) (n : Nat) : Array UPoly := Id.run do
  let x : UPoly := #[alpha, beta]
  let mut T : Array UPoly := #[#[1], x]
  for _ in [2:n+1] do
    T := T.push (UPoly.lin (UPoly.mul #[2 * alpha, 2 * beta] T[T.size - 1]!) T[T.size - 2]! 1 (-1))
  return T.extract 0 (n + 1)

/-- U_0, ..., U_n of u = alpha + beta s, as polynomials in s. -/
def chebU (alpha beta : Rat) (n : Nat) : Array UPoly := Id.run do
  let mut U : Array UPoly := #[#[1], #[2 * alpha, 2 * beta]]
  for _ in [2:n+1] do
    U := U.push (UPoly.lin (UPoly.mul #[2 * alpha, 2 * beta] U[U.size - 1]!) U[U.size - 2]! 1 (-1))
  return U.extract 0 (n + 1)

def choose : Nat → Nat → Nat
  | _, 0 => 1
  | 0, _ + 1 => 0
  | n + 1, k + 1 => choose n k + choose n (k + 1)

/-- The Bernstein coefficients of degree n on [0, 1] of a polynomial of degree <= n. -/
def bernstein (p : UPoly) (n : Nat) : Array Rat :=
  (Array.range (n + 1)).map fun i =>
    (List.range (i + 1)).foldl (fun acc j =>
      acc + ((choose i j : Nat) : Rat) / ((choose n j : Nat) : Rat) * p.getD j 0) 0

/-! ## The certificate -/

structure CountCert where
  M : Nat
  D : Nat
  r : Nat
  c1 : Rat
  c2 : Rat
  dmax : Rat
  m : Rat
  z : Array Rat
  A : Array (Array (Array Rat))

namespace CountCert

def umax (c : CountCert) : Rat := (2 * c.dmax * c.dmax - 4) / (2 * c.dmax * c.dmax)
def alpha (c : CountCert) : Rat := (4 - c.c1) / c.c2
def beta (c : CountCert) : Rat := 2 * (c.dmax - 2) / c.c2

/-- t = z^T A_0^{-1} z rounded up to a multiple of 2^-48. -/
def tVal (c : CountCert) : Rat :=
  let w := solveExact c.A[0]! c.z
  let tq := (List.range c.z.size).foldl (fun s i => s + c.z[i]! * w[i]!) 0
  ((tq * ((2 ^ 48 : Nat) : Rat)).ceil : Rat) / ((2 ^ 48 : Nat) : Rat)

/-- Check 1. -/
def positivityOk (c : CountCert) : Bool :=
  let n := c.r + 1
  let shapeOk := c.A.size == c.D + 1 && c.z.size == n &&
    c.A.all (fun a => a.size == n && a.all (fun row => row.size == n))
  let t := c.tVal
  let Z : Array (Array Rat) := ((Array.range n).map fun i => (c.A[0]![i]!).push c.z[i]!).push (c.z.push t)
  shapeOk && c.A.all isSymmetric && (List.range c.D).all (fun k => isPSD c.A[k + 1]!) &&
    isPD c.A[0]! && isPSD Z

/-- The tensor Bernstein coefficients of K on the root box, exact, flattened as
(i * (r + 1) + j) * (D + 1) + l. -/
def rootBernstein (c : CountCert) : Array Rat := Id.run do
  let r := c.r
  let D := c.D
  let Ts := (chebT c.alpha c.beta r).map fun p => bernstein p r
  let Us := (Array.range (D + 1)).map fun k =>
    bernstein (((chebU (-1) (c.umax + 1) D)[k]!).map fun x => x / ((k : Rat) + 1)) D
  let mut B : Array Rat := Array.replicate ((r + 1) * (r + 1) * (D + 1)) 0
  for k in [0:D+1] do
    for a in [0:r+1] do
      for b in [0:r+1] do
        let akab := c.A[k]![a]![b]!
        if akab != 0 then
          for i in [0:r+1] do
            let ci := akab * Ts[a]![i]!
            if ci != 0 then
              for j in [0:r+1] do
                let cij := ci * Ts[b]![j]!
                if cij != 0 then
                  for l in [0:D+1] do
                    let ix := (i * (r + 1) + j) * (D + 1) + l
                    B := B.set! ix (B[ix]! + cij * Us[k]![l]!)
  return B

/-- The Bernstein coefficients (degree 2r) of q(s) = K(d, d, 1)/2 - z . p(d), d = 2 + (dmax - 2) s. -/
def bracketBernstein (c : CountCert) : Array Rat := Id.run do
  let r := c.r
  let Tq := chebT c.alpha c.beta r
  let mut q : UPoly := #[0]
  for a in [0:r+1] do
    for b in [0:r+1] do
      let s := (List.range (c.D + 1)).foldl (fun acc k => acc + c.A[k]![a]![b]!) 0
      if s != 0 then q := UPoly.lin q (UPoly.mul Tq[a]! Tq[b]!) 1 (s / 2)
    q := UPoly.lin q Tq[a]! 1 (-c.z[a]!)
  return bernstein q (2 * r)

end CountCert

/-! ## De Casteljau at the midpoint, in exact dyadic arithmetic -/

def splitSeq (cs : Array Dy) : Array Dy × Array Dy := Id.run do
  let n := cs.size - 1
  let mut cur := cs
  let mut left : Array Dy := #[cs[0]!]
  let mut right : Array Dy := #[cs[n]!]
  for _ in [0:n] do
    let mut nxt : Array Dy := #[]
    for k in [0:cur.size - 1] do
      nxt := nxt.push (Dy.half (Dy.add cur[k]! cur[k + 1]!))
    cur := nxt
    left := left.push cur[0]!
    right := right.push cur[cur.size - 1]!
  return (left, right.reverse)

/-- Split the coefficient array (dimensions n0 x n1 x n2) at the midpoint of an axis. -/
def splitAxis (B : Array Dy) (n0 n1 n2 ax : Nat) : Array Dy × Array Dy := Id.run do
  let mut L := B
  let mut R := B
  if ax == 0 then
    for j in [0:n1] do
      for l in [0:n2] do
        let (a, b) := splitSeq ((Array.range n0).map fun i => B[(i * n1 + j) * n2 + l]!)
        for i in [0:n0] do
          L := L.set! ((i * n1 + j) * n2 + l) a[i]!
          R := R.set! ((i * n1 + j) * n2 + l) b[i]!
  else if ax == 1 then
    for i in [0:n0] do
      for l in [0:n2] do
        let (a, b) := splitSeq ((Array.range n1).map fun j => B[(i * n1 + j) * n2 + l]!)
        for j in [0:n1] do
          L := L.set! ((i * n1 + j) * n2 + l) a[j]!
          R := R.set! ((i * n1 + j) * n2 + l) b[j]!
  else
    for i in [0:n0] do
      for j in [0:n1] do
        let (a, b) := splitSeq ((Array.range n2).map fun l => B[(i * n1 + j) * n2 + l]!)
        for l in [0:n2] do
          L := L.set! ((i * n1 + j) * n2 + l) a[l]!
          R := R.set! ((i * n1 + j) * n2 + l) b[l]!
  return (L, R)

def maxDy (B : Array Dy) : Dy := B.foldl Dy.max (B[0]!)

/-! ## Check 2: the pair inequality -/

structure PBox where
  i1 : Nat
  n1 : Nat
  i2 : Nat
  n2 : Nat
  i3 : Nat
  n3 : Nat
  B : Array Dy
  depth : Nat

structure PStat where
  closed : Nat := 0
  dropped : Nat := 0
  piCalls : Nat := 0
  status : Nat := 0     -- 0 done, 1 a box reached the depth limit, 2 fuel exhausted

abbrev Key := Int × Nat × Int × Nat × Int × Nat
def ratKey (a b c : Rat) : Key := (a.num, a.den, b.num, b.den, c.num, c.den)

structure PCtx where
  c : CountCert
  um : Rat           -- umax
  wU : Rat           -- the weight of the u side: (umax + 1)/(dmax - 2)/3

def PCtx.dval (x : PCtx) (i n : Nat) : Rat := 2 + (x.c.dmax - 2) * (i : Rat) / ((2 ^ n : Nat) : Rat)
def PCtx.uval (x : PCtx) (i n : Nat) : Rat := -1 + (x.um + 1) * (i : Rat) / ((2 ^ n : Nat) : Rat)
def amaxq (d1 d2 : Rat) : Rat := (d1 * d1 + d2 * d2 - 4) / (2 * d1 * d2)

/-- Pi(d1/2, d2/2, u) in floating point, as truncated_search.pair computes it.  It
only decides whether the rigorous lower bound piLower is worth computing on a box;
no box is closed on its strength. -/
def piFloat (d1 d2 u : Float) : Float :=
  let R := Float.sqrt 1.5
  let clip (a lo hi : Float) : Float := if a < lo then lo else if a > hi then hi else a
  let gF (h x : Float) : Float :=
    let tx := Float.tan x
    2.25 * x - 3 * h * h * tx + h * h * h * h * (tx + tx * tx * tx / 3)
  let h1 := d1 / 2
  let h2 := d2 / 2
  let g := Float.acos (clip u (-1) 1)
  let a1 := Float.acos (clip (h1 / R) (-1) 1)
  let a2 := Float.acos (clip (h2 / R) (-1) 1)
  let L := if -a1 > g - a2 then -a1 else g - a2
  let U := if a1 < g + a2 then a1 else g + a2
  let sg := Float.sin g
  let pc := clip (Float.atan2 (h2 - h1 * Float.cos g) (h1 * (if sg > 1e-15 then sg else 1e-15))) L U
  if L < U then 3.141592653589793 / 4 * (gF h2 (pc - g) - gF h2 (L - g) + gF h1 U - gF h1 pc) else 0

def ratToFloat (q : Rat) : Float := Float.ofInt q.num / q.den.toFloat

/-- A dyadic number in floating point (only used by the filter above). -/
def Dy.toFloat (a : Dy) : Float :=
  if a.e ≤ 60 then Float.ofInt a.n / (2 ^ a.e : Nat).toFloat
  else Float.ofInt (Int.fdiv a.n (Rd.pow2 (a.e - 60))) / (2 ^ 60 : Nat).toFloat

def runPairs (x : PCtx) (r D : Nat) :
    Nat → List PBox → PStat → Std.HashMap Key Dy → PStat
  | 0, _, st, _ => { st with status := 2 }
  | _, [], st, _ => st
  | fuel + 1, b :: rest, st, memo =>
    let dlo := x.dval b.i1 b.n1
    let dhi := x.dval (b.i1 + 1) b.n1
    let ehi := x.dval (b.i2 + 1) b.n2
    let ulo := x.uval b.i3 b.n3
    if dlo > ehi || ulo > amaxq dhi ehi then
      runPairs x r D fuel rest { st with dropped := st.dropped + 1 } memo
    else
      let kub := maxDy b.B
      if Dy.le kub Dy.zero then
        runPairs x r D fuel rest { st with closed := st.closed + 1 } memo
      else
        let pf := piFloat (ratToFloat dhi) (ratToFloat ehi) (ratToFloat ulo)
        let tryPi := kub.toFloat ≤ pf - 1e-9
        let key := ratKey dhi ehi ulo
        let (pl, memo', calls) :=
          if !tryPi then (Dy.zero, memo, st.piCalls) else
          match memo.get? key with
          | some v => (v, memo, st.piCalls)
          | none => let v := piLower dhi ehi ulo; (v, memo.insert key v, st.piCalls + 1)
        let st := { st with piCalls := calls }
        if tryPi && Dy.le kub pl then
          runPairs x r D fuel rest { st with closed := st.closed + 1 } memo'
        else if b.depth ≥ 80 then
          { st with status := 1 }
        else
          let w0 : Rat := 1 / ((2 ^ b.n1 : Nat) : Rat)
          let w1 : Rat := 1 / ((2 ^ b.n2 : Nat) : Rat)
          let w2 : Rat := x.wU / ((2 ^ b.n3 : Nat) : Rat)
          let ax := if w0 ≥ w1 && w0 ≥ w2 then 0 else if w1 ≥ w2 then 1 else 2
          let (L, R) := splitAxis b.B (r + 1) (r + 1) (D + 1) ax
          let d := b.depth + 1
          let kids : List PBox :=
            if ax == 0 then
              [⟨2 * b.i1, b.n1 + 1, b.i2, b.n2, b.i3, b.n3, L, d⟩, ⟨2 * b.i1 + 1, b.n1 + 1, b.i2, b.n2, b.i3, b.n3, R, d⟩]
            else if ax == 1 then
              [⟨b.i1, b.n1, 2 * b.i2, b.n2 + 1, b.i3, b.n3, L, d⟩, ⟨b.i1, b.n1, 2 * b.i2 + 1, b.n2 + 1, b.i3, b.n3, R, d⟩]
            else
              [⟨b.i1, b.n1, b.i2, b.n2, 2 * b.i3, b.n3 + 1, L, d⟩, ⟨b.i1, b.n1, b.i2, b.n2, 2 * b.i3 + 1, b.n3 + 1, R, d⟩]
          runPairs x r D fuel (kids ++ rest) st memo'

def CountCert.pairStat (c : CountCert) (fuel : Nat) : PStat :=
  let x : PCtx := ⟨c, c.umax, (c.umax + 1) / (c.dmax - 2) / 3⟩
  let B0 := c.rootBernstein.map fun q => Dy.ceilRat q 320
  runPairs x c.r c.D fuel [⟨0, 0, 0, 0, 0, 0, B0, 0⟩] {} {}

/-- Check 2, with dmax^2 > 6 so that the boxes cover every distance below sqrt 6. -/
def CountCert.pairsOk (c : CountCert) : Bool :=
  c.dmax * c.dmax > 6 && (c.pairStat 20000000).status == 0

/-! ## Check 3: the bracket -/

structure BStat where
  count : Nat := 0
  status : Nat := 0

def runBracket (c : CountCert) : Nat → List (Nat × Nat × Array Dy × Nat) → BStat → BStat
  | 0, _, st => { st with status := 2 }
  | _, [], st => st
  | fuel + 1, (i, n, B, depth) :: rest, st =>
    let dlo := 2 + (c.dmax - 2) * (i : Rat) / ((2 ^ n : Nat) : Rat)
    let bound := Dy.toRat (capS dlo).hi + Dy.toRat (maxDy B)
    if bound < c.m then runBracket c fuel rest { st with count := st.count + 1 }
    else if depth ≥ 60 then { st with status := 1 }
    else
      let (L, R) := splitSeq B
      runBracket c fuel ((2 * i, n + 1, L, depth + 1) :: (2 * i + 1, n + 1, R, depth + 1) :: rest) st

def CountCert.bracketStat (c : CountCert) : BStat :=
  runBracket c 100000 [(0, 0, c.bracketBernstein.map (fun q => Dy.ceilRat q 320), 0)] {}

def CountCert.bracketOk (c : CountCert) : Bool := c.bracketStat.status == 0

/-! ## Check 4: the conclusion -/

/-- m < 0 and M m + t/2 < 9 pi^2/8 - 8, the right side bounded below in intervals. -/
def CountCert.finalOk (c : CountCert) : Bool :=
  let target := RI.sub (RI.scaleRat (9/8) (RI.mul piI piI)) (RI.ofNat 8)
  c.m < 0 && (c.M : Rat) * c.m + c.tVal / 2 < Dy.toRat target.lo

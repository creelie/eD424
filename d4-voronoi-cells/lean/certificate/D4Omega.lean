/-
D4Omega.lean: the transcendental inputs of the certificate checks, computed
inside Lean.

The branch and bounds of D4CertDomain (thm:certificate, region I of thm:m23)
and D4LabelledDomain (regions II_s and II_f) take omega, omega' and omega''
from tables, and the labelled check takes the slab constant and the cell
integrals of A(tau) as data; D4Certificate.lean compares the bound of the
certificate with 8 - A_*.  Here all of these are computed from their closed
forms in outward-rounded dyadic interval arithmetic (256 bits), with square
roots from Nat.sqrt and arctan from its alternating series:

  omega(u)   = 4 pi [ (9/32) arctan((t* - tau)/(1 + t* tau)) - (4 tau^3 - 24 tau + 11 sqrt2)/96 ],
  omega'(u)  = (pi/16) (3u - 1)^2 / ((1 + u)^2 sqrt(1 - u^2)),
  omega''(u) = (pi/16) (3u - 1) (1 + u)^(-5/2) (1 - u)^(-1/2) [6 - (3u - 1)(2 - 3u)/(1 - u^2)],
  A_*        = 9 pi^2 / 8 - 207 pi r_* / 8 + 253 pi / (12 sqrt2),
  S(d)       = (4 pi / 3) [ (27/32)(pi/2) - (h/8)(5 R^2 - 2h^2)(R^2 - h^2)^(1/2) - (27/32) arctan(h / (R^2 - h^2)^(1/2)) ],

for 1/3 <= u < 1, tau = ((1 - u)/(1 + u))^(1/2), t* = tan r_* = 1/sqrt2, h = d/2,
R^2 = 3/2.  They are the closed forms of the paper (the lens integral of
lem:pair-closed and its derivatives, and the cap of B(sqrt(3/2))), simplified;
multi_cap/certificate_check.py evaluates the unsimplified forms, and the two
agree to 1e-14 (checked with mpmath; the derivative identity symbolically).
The tables are on the dyadic grid of step 2^-16 from the first grid point above
1/3; the bound on |omega''| is taken over 4096 subintervals.
-/
import D4LabelledDomain

namespace Rd
def P : Nat := 256
def pow2 (k : Nat) : Int := Int.ofNat (Nat.shiftLeft 1 k)
/-- Round down / up to at most P fractional bits. -/
def down (a : Dy) : Dy := if a.e ≤ P then a else ⟨Int.fdiv a.n (pow2 (a.e - P)), P⟩
def up (a : Dy) : Dy := if a.e ≤ P then a else ⟨-(Int.fdiv (-a.n) (pow2 (a.e - P))), P⟩
def toRat (a : Dy) : Rat := (a.n : Rat) / ((2 ^ a.e : Nat) : Rat)
end Rd

/-- Closed intervals with dyadic ends, rounded outward. -/
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
def add (x y : RI) : RI := ⟨down (x.lo.add y.lo), up (x.hi.add y.hi)⟩
def neg (x : RI) : RI := ⟨x.hi.neg, x.lo.neg⟩
def sub (x y : RI) : RI := add x (neg y)
def mul (x y : RI) : RI :=
  let p1 := x.lo.mul y.lo; let p2 := x.lo.mul y.hi; let p3 := x.hi.mul y.lo; let p4 := x.hi.mul y.hi
  ⟨down (Dy.min (Dy.min p1 p2) (Dy.min p3 p4)), up (Dy.max (Dy.max p1 p2) (Dy.max p3 p4))⟩
def scaleRat (q : Rat) (x : RI) : RI := mul (ofRat q) x

/-- a / b rounded down and up, for b > 0. -/
def divDown (a b : Dy) : Dy := ⟨Int.fdiv (a.n * pow2 (b.e + P)) (b.n * pow2 a.e), P⟩
def divUp (a b : Dy) : Dy := ⟨-(Int.fdiv (-(a.n * pow2 (b.e + P))) (b.n * pow2 a.e)), P⟩

def isPos (x : RI) : Bool := x.lo.n > 0
/-- x / y for y > 0; a huge interval otherwise, which no check passes. -/
def div (x y : RI) : RI :=
  if !isPos y then ⟨⟨-(pow2 400), 0⟩, ⟨pow2 400, 0⟩⟩ else
  let c := [divDown x.lo y.lo, divDown x.lo y.hi, divDown x.hi y.lo, divDown x.hi y.hi]
  let d := [divUp x.lo y.lo, divUp x.lo y.hi, divUp x.hi y.lo, divUp x.hi y.hi]
  ⟨c.foldl Dy.min (c.getD 0 Dy.zero), d.foldl Dy.max (d.getD 0 Dy.zero)⟩

/-- Square roots of a >= 0 rounded down and up (a is first rounded to 2P bits). -/
def sqrtDown (a : Dy) : Dy :=
  if a.n ≤ 0 then Dy.zero else
  let m : Nat := (if a.e ≤ 2 * P then a.n * pow2 (2 * P - a.e) else Int.fdiv a.n (pow2 (a.e - 2 * P))).toNat
  ⟨Nat.sqrt m, P⟩
def sqrtUp (a : Dy) : Dy :=
  if a.n ≤ 0 then Dy.zero else
  let m : Nat := (if a.e ≤ 2 * P then a.n * pow2 (2 * P - a.e) else -(Int.fdiv (-a.n) (pow2 (a.e - 2 * P)))).toNat
  let r := Nat.sqrt m
  ⟨if r * r ≥ m then r else r + 1, P⟩
def sqrt (x : RI) : RI := ⟨sqrtDown x.lo, sqrtUp x.hi⟩

def absHi (x : RI) : Dy := Dy.max x.lo.abs x.hi.abs
def pow (x : RI) : Nat → RI
  | 0 => pt Dy.one
  | n + 1 => mul (pow x n) x
def hull (x y : RI) : RI := ⟨Dy.min x.lo y.lo, Dy.max x.hi y.hi⟩
def loRat (x : RI) : Rat := toRat x.lo
def hiRat (x : RI) : Rat := toRat x.hi
end RI

/-! ## arctan and pi -/

/-- arctan on an interval x with |x| <= 1/2, by 120 terms of the series evaluated in
interval arithmetic (an enclosure of the partial sums over x), with the remainder
|x|^241 / 241 added on both sides. -/
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

/-- pi by Machin's formula, 16 arctan(1/5) - 4 arctan(1/239). -/
def piI0 : RI :=
  RI.sub (RI.scaleRat 16 (atanSeries (RI.ofRat (1/5)))) (RI.scaleRat 4 (atanSeries (RI.ofRat (1/239))))

/-- arctan at a point a >= 0: for a > 2, pi/2 - arctan(1/a); for 1/2 < a <= 2,
2 arctan(a / (1 + sqrt(1 + a^2))); for a <= 1/2, the series.  The reductions are
monotone, so they are applied to the ends of the intervals they produce. -/
def atanNonneg : Nat → Dy → RI
  | 0, _ => ⟨⟨-(Rd.pow2 400), 0⟩, ⟨Rd.pow2 400, 0⟩⟩
  | fuel + 1, a =>
    if Dy.lt ⟨2, 0⟩ a then
      let inv := RI.div (RI.pt Dy.one) (RI.pt a)
      let t := RI.hull (atanNonneg fuel inv.lo) (atanNonneg fuel inv.hi)
      RI.sub (RI.scaleRat (1/2) piI0) t
    else if Dy.lt ⟨1, 1⟩ a then
      let b := RI.div (RI.pt a) (RI.add (RI.pt Dy.one) (RI.sqrt (RI.add (RI.pt Dy.one) (RI.mul (RI.pt a) (RI.pt a)))))
      let t := RI.hull (atanNonneg fuel b.lo) (atanNonneg fuel b.hi)
      RI.add t t
    else atanSeries (RI.pt a)

def piI : RI := piI0

/-- arctan at a point, any sign. -/
def atanPt (a : Dy) : RI := if a.n < 0 then RI.neg (atanNonneg 4 a.neg) else atanNonneg 4 a

/-- arctan on an interval, by monotonicity. -/
def atanI (x : RI) : RI := ⟨(atanPt x.lo).lo, (atanPt x.hi).hi⟩

/-! ## omega and its derivatives -/

def sqrtHalf : RI := RI.sqrt (RI.ofRat (1/2))
def sqrt2 : RI := RI.sqrt (RI.ofNat 2)

/-- omega(u) for 1/3 <= u < 1. -/
def omegaI (u : RI) : RI :=
  let one := RI.ofNat 1
  let tau := RI.sqrt (RI.div (RI.sub one u) (RI.add one u))
  let x := RI.div (RI.sub sqrtHalf tau) (RI.add one (RI.mul sqrtHalf tau))
  let alg := RI.scaleRat (-1/96)
    (RI.add (RI.sub (RI.scaleRat 4 (RI.pow tau 3)) (RI.scaleRat 24 tau)) (RI.scaleRat 11 sqrt2))
  RI.mul (RI.scaleRat 4 piI) (RI.add (RI.scaleRat (9/32) (atanI x)) alg)

/-- omega'(u) = (pi/16) (3u - 1)^2 / ((1 + u)^2 sqrt(1 - u^2)). -/
def omega1I (u : RI) : RI :=
  let one := RI.ofNat 1
  let a := RI.sub (RI.scaleRat 3 u) one
  let w := RI.add one u
  RI.div (RI.mul (RI.scaleRat (1/16) piI) (RI.mul a a))
    (RI.mul (RI.mul w w) (RI.sqrt (RI.sub one (RI.mul u u))))

/-- omega''(u) = (pi/16) (3u - 1) (1 + u)^(-5/2) (1 - u)^(-1/2) [6 - (3u - 1)(2 - 3u)/(1 - u^2)]. -/
def omega2I (u : RI) : RI :=
  let one := RI.ofNat 1
  let a := RI.sub (RI.scaleRat 3 u) one
  let w := RI.add one u
  let v := RI.sub one u
  let br := RI.sub (RI.ofNat 6) (RI.div (RI.mul a (RI.sub (RI.ofNat 2) (RI.scaleRat 3 u))) (RI.mul w v))
  RI.div (RI.mul (RI.mul (RI.scaleRat (1/16) piI) a) br)
    (RI.mul (RI.mul (RI.mul w w) (RI.sqrt w)) (RI.sqrt v))

/-! ## The tables on the grid of step 2^-16 -/

def gridE : Nat := 16
def i0 : Nat := 21846            -- 21846 / 2^16 = 0.333343... > 1/3 > 21845 / 2^16

def gridPt (i : Nat) : Dy := ⟨(i0 + i : Nat), gridE⟩

/-- The table from the first grid point above 1/3 to the first grid point at or above umax,
with |omega''| bounded over 4096 subintervals of [1/3, that point]. -/
def leanTab (umax : Rat) : OTab := Id.run do
  let n : Nat := ((umax * ((2 ^ gridE : Nat) : Rat)).ceil.toNat) - i0 + 1
  let mut us : Array Dy := #[]
  let mut olow : Array Dy := #[]
  let mut d1lo : Array Dy := #[]
  let mut d1hi : Array Dy := #[]
  for i in [0:n] do
    let u := gridPt i
    let w := omegaI (RI.pt u)
    let w1 := omega1I (RI.pt u)
    us := us.push u
    olow := olow.push (Dy.max w.lo Dy.zero)
    d1lo := d1lo.push w1.lo
    d1hi := d1hi.push w1.hi
  let top := Rd.toRat (gridPt (n - 1))
  let mut m2 : Dy := Dy.zero
  for j in [0:4096] do
    let a := RI.ofRat (1/3 + (top - 1/3) * (j : Rat) / 4096)
    let b := RI.ofRat (1/3 + (top - 1/3) * ((j + 1 : Nat) : Rat) / 4096)
    m2 := Dy.max m2 (RI.absHi (omega2I ⟨a.lo, b.hi⟩))
  return ⟨us, olow, d1lo, d1hi, ⟨1, gridE⟩, m2⟩

/-- For region I (u <= 1/2) and for the labelled regions (u <= a_D). -/
def leanTabI : OTab := leanTab (1 / 2)
def leanTabL : OTab := leanTab aDtop.toRat

/-! ## A_*, the cap S(d), and the constants of the labelled certificate -/

def rStar : RI := atanI sqrtHalf

/-- A_* = 9 pi^2 / 8 - 207 pi r_* / 8 + 253 pi / (12 sqrt2). -/
def aStar : RI :=
  RI.add (RI.sub (RI.scaleRat (9/8) (RI.mul piI piI)) (RI.scaleRat (207/8) (RI.mul piI rStar)))
    (RI.div (RI.scaleRat (253/12) piI) sqrt2)

/-- S(d) for 2 <= d <= sqrt 6, h = d/2 given as an interval. -/
def capS (h : RI) : RI :=
  let r2 := RI.ofRat (3/2)
  let s := RI.sub r2 (RI.mul h h)
  let sq := RI.sqrt s
  let asin := atanI (RI.div h sq)
  let inner := RI.add (RI.mul (RI.scaleRat (1/8) h) (RI.mul (RI.sub (RI.scaleRat 5 r2) (RI.scaleRat 2 (RI.mul h h))) sq))
                      (RI.scaleRat (27/32) asin)
  RI.mul (RI.scaleRat (4/3) piI) (RI.sub (RI.scaleRat (27/64) piI) inner)

/-- s(d) = S(2) - S(d). -/
def smallS (d : Rat) : RI := RI.sub (capS (RI.ofNat 1)) (capS (RI.ofRat (d / 2)))

/-- The cell integrals ds_k = S(2 tau_k) - S(2 tau_{k+1}). -/
def dsI (k : Nat) : RI := RI.sub (capS (RI.ofRat (tauK k))) (capS (RI.ofRat (tauK (k + 1))))

/-- The constants: A_* against the bound of the certificate of thm:certificate (at least
92.8555703 in the units of the solver, D4Certificate.lean); step 2 of thm:m23,
s(D) > 8 - A_*, a_D / (1 - (D/2) a_D) < 2 (the share decreases in both heights),
fr(1, 1/2) < 1/22, and the slab constant of the data at most
4000 r kappa, r = 1/11 - 2 fr(1, 51/100), kappa = s(2.04) / 0.04; and the enclosures
of the 256 cell integrals of the data containing those computed here. -/
def labelledConstantsOk : Bool :=
  let aLo := aStar.loRat
  let sD := (smallS Dd).loRat
  let r := 1 / 11 - 2 * frHi 1 (51 / 100)
  let kappa := (smallS (204 / 100)).loRat / (4 / 100)
  aLo > 8 - 928555703 / 10000000000 &&
  sD + aLo > 8 &&
  amaxR Dd Dd / (1 - Dd / 2 * amaxR Dd Dd) < 2 &&
  frHi 1 (1 / 2) < 1 / 22 &&
  r > 0 && cSlab.toRat ≤ 4000 * r * kappa &&
  (List.range KC).all fun k =>
    let d := dsI k
    Dy.le (dsLo[k]!) d.lo && Dy.le d.hi (dsHi[k]!)

/-- The grids increase in steps of du, reach the top of the domains, and every entry is finite. -/
def tableOk (t : OTab) (umax : Dy) : Bool :=
  t.us.size == t.olow.size && t.us.size == t.d1lo.size && t.us.size == t.d1hi.size &&
  (List.range (t.us.size - 1)).all (fun i => Dy.le ((t.us[i + 1]!).sub (t.us[i]!)) t.du &&
                                             Dy.lt (t.us[i]!) (t.us[i + 1]!)) &&
  Dy.le umax (t.us[t.us.size - 1]!) && Dy.lt Dy.zero t.m2 &&
  (List.range t.us.size).all fun i => Dy.le (t.d1lo[i]!) (t.d1hi[i]!)

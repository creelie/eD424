/-
D4NearContact.lean

The exact arithmetic behind the explicit constant epsilon_0 = 4e-26 of the
near-contact theorem (thm:near-contact of the paper; multi_cap/explicit_eps0.py).

  B.  The two-point polynomial p_2 of the certificate of de Laat, Leijenhorst and
      de Muinck Keizer, exact from their data (llm24_p2.txt), is
      (u+1)(u+1/2)^2 u^2 (u-1/2) q(u) with q of degree 10, and -q >= 2.62e-4 on
      [-1, 1/2] (branch and bound, second-order Taylor form, exact rationals).
      With f(u) = (u+1)(u+1/2)^2 u^2 (1/2-u), f >= 1.12e-8 at distance at least
      delta = 3e-4 from -1, -1/2, 0, 1/2 (same branch and bound), so there
      p_2 = f (-q) >= 2.62e-4 * 1.12e-8.
  C.  The robust root-lattice step at delta: for k <= 5,
      (row + 2 delta sqrt k)^k - row^k < 1 with row = (8 + 3 (1 + 2 delta)^2)^(1/2),
      and d(W) <= 2 D / (6 - D)^(1/2) + sqrt 24 D / 6 <= 0.0112709 < 1/48, D = 23 delta.
  E.  With |p_2'| <= L2 on [1/2, 1/2 + 1/1000] (computed here) and the bounds
      B_3 <= 8.75e7, B_4 <= 7.34e9 of part A (from explicit_eps0.py, ball
      arithmetic on the deposited certificate; the two inputs not computed here),
      E(m, kappa) = C(m,2) kappa L2 + C(m,3) a B_3 + C(m,4) a B_4, a = kappa (3/2 + kappa),
      satisfies E(24, kappa*) < 2.62e-4 * 1.12e-8 and E(25, kappa*) < 1 at
      kappa* = 2e-26; kappa* < delta; and 1/2 - 2/(2 + epsilon_0)^2 <= kappa* for
      epsilon_0 = 2 kappa*.
  D.  The facet-by-facet bracket of step 2 of the proof, in exact rationals as the
      script computes it: the constants of estimate (a), the bracket positive at
      S = 24 epsilon_0, at least 1/10 at S = 4e-3 with Theta <= 0.014 there, and
      decreasing on the grid of step 1e-5 up to 4e-3.

The finite checks are settled by native_decide.  What makes these numbers a
proof (the sums of squares of part A, Hadamard's inequality behind part C, the
geometry behind part D) is in the paper and is not formalised; Sturm's theorem
for q is in D4InnerProducts.lean.

No `sorry`, no Mathlib; compiles against a bare Lean 4 toolchain.
-/

set_option maxRecDepth 10000

namespace D4NearContact

/-- p_2 = (sum_i p2num[i] u^i) / p2den, lowest degree first. -/
def p2num : List Int := [
  @@P2NUM@@]

def p2den : Nat := @@P2DEN@@

/-! ## Polynomials and intervals over the rationals -/

abbrev Poly := List Rat

def eval (p : Poly) (x : Rat) : Rat := p.foldr (fun c acc => c + x * acc) 0

def deriv (p : Poly) : Poly :=
  match p with
  | [] => []
  | _ :: cs => (List.range cs.length).zipWith (fun i c => ((i : Nat) + 1 : Rat) * c) cs

def pmul (p q : Poly) : Poly :=
  p.zipIdx.foldl (fun acc (a, i) =>
    let term := List.replicate i (0 : Rat) ++ q.map (a * ·)
    let n := Nat.max acc.length term.length
    (List.range n).map fun k => acc.getD k 0 + term.getD k 0) []

/-- Synthetic division by (x - a): the quotient and the remainder p(a). -/
def divLinear (p : Poly) (a : Rat) : Poly × Rat :=
  let rev := p.reverse
  let rec go : List Rat → Rat → List Rat
    | [], _ => []
    | c :: cs, carry => let b := c + a * carry; b :: go cs b
  let out := go rev 0
  (out.dropLast.reverse, out.getLastD 0)

def rabs (x : Rat) : Rat := if x < 0 then -x else x
def rmax (x y : Rat) : Rat := if x ≤ y then y else x
def rmin (x y : Rat) : Rat := if x ≤ y then x else y

structure RI where
  lo : Rat
  hi : Rat

def RI.add (x y : RI) : RI := ⟨x.lo + y.lo, x.hi + y.hi⟩
def RI.mul (x y : RI) : RI :=
  let a := x.lo * y.lo; let b := x.lo * y.hi; let c := x.hi * y.lo; let d := x.hi * y.hi
  ⟨rmin (rmin a b) (rmin c d), rmax (rmax a b) (rmax c d)⟩
def RI.absHi (x : RI) : Rat := rmax (rabs x.lo) (rabs x.hi)

/-- An enclosure of p on [a, b], by Horner's scheme in interval arithmetic. -/
def evalIv (p : Poly) (x : RI) : RI := p.foldr (fun c acc => RI.add ⟨c, c⟩ (RI.mul x acc)) ⟨0, 0⟩

/-! ## Branch and bound for p >= target on an interval -/

/-- A lower bound of p on [a, b]: second-order Taylor form about the midpoint. -/
def lowerOn (p p1 p2 : Poly) (a b : Rat) : Rat :=
  let m := (a + b) / 2
  let r := (b - a) / 2
  eval p m - rabs (eval p1 m) * r - (evalIv p2 ⟨a, b⟩).absHi * r * r / 2

/-- (status, intervals): status 0 when p >= target was shown on every piece,
1 when a piece of width below 2^-40 was undecided, 2 when the fuel ran out. -/
def bnb (p p1 p2 : Poly) (target : Rat) (done : Nat) : Nat → List (Rat × Rat) → Nat × Nat
  | 0, _ => (2, done)
  | _, [] => (0, done)
  | fuel + 1, (a, b) :: rest =>
    if lowerOn p p1 p2 a b ≥ target then bnb p p1 p2 target (done + 1) fuel rest
    else if b - a < 1 / 2 ^ 40 then (1, done)
    else
      let m := (a + b) / 2
      bnb p p1 p2 target (done + 1) fuel ((a, m) :: (m, b) :: rest)

def atLeastOn (p : Poly) (target : Rat) (pieces : List (Rat × Rat)) : Nat × Nat :=
  let p1 := deriv p
  bnb p p1 (deriv p1) target 0 1000000 pieces

/-! ## B. The two-point polynomial -/

def p2 : Poly := p2num.map fun (c : Int) => (c : Rat) / (p2den : Rat)

/-- Division by u + 1, (u + 1/2)^2, u^2 and u - 1/2, with the six remainders. -/
def steps : List (Poly × Rat) :=
  [-1, -1/2, -1/2, 0, 0, 1/2].foldl (fun acc a =>
    let prev := (acc.getLastD (p2, 0)).1
    acc ++ [divLinear prev a]) []

def q : Poly := (steps.getLastD ([], 0)).1

theorem division_exact : (p2.length == 17 && steps.all (fun s => s.2 == 0) && q.length == 11) = true := by
  native_decide

def negq : Poly := q.map (- ·)

/-- q_min = 2.62e-4 -/
def qmin : Rat := 262 / 1000000

theorem minus_q_lower_bound : (atLeastOn negq qmin [(-1, 1/2)]).1 = 0 := by native_decide

def delta : Rat := 3 / 10000

/-- f(u) = (u+1)(u+1/2)^2 u^2 (1/2-u) -/
def fpoly : Poly := pmul (pmul (pmul (pmul [1, 1] [1/2, 1]) [1/2, 1]) [0, 0, 1]) [1/2, -1]

/-- f = -(u+1)(u+1/2)^2 u^2 (u-1/2), so that p_2 = f (-q). -/
theorem p2_factorisation : (pmul fpoly negq == p2) = true := by native_decide

def fmin : Rat := 112 / 10000000000

theorem f_lower_bound :
    (atLeastOn fpoly fmin [(-1 + delta, -1/2 - delta), (-1/2 + delta, -delta), (delta, 1/2 - delta)]).1 = 0 := by
  native_decide

/-- the bound on p_2 at distance >= delta from the four zeros -/
def need : Rat := qmin * fmin

/-! ## C. The robust root-lattice step -/

/-- A rational >= sqrt y (y >= 0), with its square checked. -/
def sqrtUp (y : Rat) : Rat :=
  let p : Nat := 60
  let n : Nat := (y * ((4 ^ p : Nat) : Rat)).ceil.toNat
  let r := Nat.sqrt n
  let r := if r * r ≥ n then r else r + 1
  let s : Rat := (r : Rat) / ((2 ^ p : Nat) : Rat)
  if s * s ≥ y then s else s + 1

/-- A rational <= sqrt y (y >= 0), with its square checked. -/
def sqrtDown (y : Rat) : Rat :=
  let p : Nat := 60
  let n : Nat := (y * ((4 ^ p : Nat) : Rat)).floor.toNat
  let s : Rat := (Nat.sqrt n : Rat) / ((2 ^ p : Nat) : Rat)
  if s * s ≤ y then s else 0

def rowHi : Rat := sqrtUp (8 + 3 * (1 + 2 * delta) ^ 2)

/-- (x + e)^k - x^k increases in x for e >= 0, so the bound at rowHi covers row. -/
theorem minors_within_one :
    (([1, 2, 3, 4, 5] : List Nat).all fun k =>
      (rowHi + 2 * delta * sqrtUp (k : Rat)) ^ k - rowHi ^ k < 1) = true := by native_decide

def Dn : Rat := 23 * delta

theorem dW_bound :
    (2 * Dn / sqrtDown (6 - Dn) + sqrtUp 24 * Dn / 6 ≤ 112709 / 10000000 &&
     (112709 / 10000000 : Rat) < 1 / 48) = true := by native_decide

/-! ## E. The assembly -/

/-- |p_2'| on [1/2, 1/2 + 1/1000] -/
def L2 : Rat := (evalIv (deriv p2) ⟨1/2, 1/2 + 1/1000⟩).absHi

def B3 : Rat := 875 * 10 ^ 5
def B4 : Rat := 734 * 10 ^ 7

def choose : Nat → Nat → Nat
  | _, 0 => 1
  | 0, _ + 1 => 0
  | n + 1, k + 1 => choose n k + choose n (k + 1)

def Ebound (m : Nat) (kap : Rat) : Rat :=
  let a := kap * (3 / 2 + kap)
  (choose m 2 : Rat) * kap * L2 + (choose m 3 : Rat) * a * B3 + (choose m 4 : Rat) * a * B4

def kstar : Rat := 2 / 10 ^ 26
def eps0 : Rat := 2 * kstar

theorem assembly :
    (L2 ≤ 22573 / 100000 && Ebound 24 kstar < need && Ebound 25 kstar < 1 && kstar < delta &&
     1 / 2 - 2 / (2 + eps0) ^ 2 ≤ kstar && eps0 = 4 / 10 ^ 26) = true := by native_decide

/-! ## D. The bracket of step 2 -/

def S11 : Rat := 33166248 / 10000000
def RIG_L : Rat := S11 / 4 + 12991 / 10000
def RIG_M : Rat := 6 * S11 / 8 + 6 * (10826 / 10000) + 1 / 2

/-- The rational square-root bounds behind RIG_L and RIG_M. -/
theorem rigidity_roots :
    (S11 * S11 > 11 && (12991 / 10000 : Rat) ^ 2 > 27 / 16 && (10826 / 10000 : Rat) ^ 2 > 75 / 64) = true := by
  native_decide

/-- estimate (a): ||eps|| <= RIG_L (1 + delta) S + RIG_M ||eps||^2, first with ||eps|| <= 1/48, then fed back once;
none if the first pass does not stay below 1/48. -/
def rigidity (S : Rat) : Option Rat :=
  let x1 := RIG_L * (1 + S) * S / (1 - RIG_M / 48)
  if x1 ≤ 1 / 48 then some (RIG_L * (1 + S) * S / (1 - RIG_M * x1)) else none

/-- The bracket B with vol - 8 >= S B, and alpha > 2/3; as explicit_eps0.local_bracket. -/
def bracket (S : Rat) : Option (Rat × Bool) := do
  let E ← rigidity S
  let s2 : Rat := 14143 / 10000
  let s3 : Rat := 17321 / 10000
  let s3l : Rat := 1732 / 1000
  let s5 : Rat := 22361 / 10000
  let s24 : Rat := 4899 / 1000
  let d := S
  let e := E
  let r1 := s2 * (1 + d / 2) / (1 - s2 * e)
  let eta1 := d / 2 + r1 * e
  let eta2 := eta1 + d / 2 + r1 * e
  let R := 1 + 2 * eta2
  let Rp := R + e / 2
  let c3 := (1 / 2 + e) / (1 - e * e / 2)
  let k1 := 2 / (s3l * (1 - e))
  let Dmax := Rp * e + d / 2 + c3 * (d / 2 + Rp * e)
  let thmax := k1 * Dmax
  let c0 := 1 / s3 - thmax
  let a_in := 3 * s3 / 4 * (1 - c0 * c0)
  let lam := 1 + s3 * thmax
  let a_out := 3 * s3 / 4 * (lam * lam - 1 / 3)
  let gbar := 1 / (1 - e * e / 2) ^ 2
  let A1 := k1 * 8 * (Rp + c3 * Rp)
  let B1 := k1 * 4 * (1 + c3)
  let nd := S
  let s_e := A1 * E * E + B1 * E * nd
  let s_d := A1 * nd * E + B1 * nd * nd
  let loss_v := 1 / 2 * a_in * (1 / 2 * s_d + 1 / 2 * e * s_e)
  let loss_m := 1 / 2 * gbar * (a_in + R * a_out) * s_e
  let Dperp := s5 * e + e * e / 2 + d / 2 + e * (d / 2 + e * e / 2 + e) / (1 - e * e / 2)
  let loss_c := 1 / 4 * (1 / 2 * S + 1 / 2 * E * E + gbar * s24 * E) * 4 * Dperp ^ 3
  let Theta := e + d
  let hull := 24 * 2 * 4 * (3 / 2 + 142 / 100) ^ 4 * Theta ^ 4
  let alpha := 1 / 2 * gbar * (a_in + R * a_out) * A1 + 1 / 4 * a_in * e * A1
  let total := 2 / 3 * S + 2 / 3 * E * E - loss_v - loss_m - loss_c - hull
  return (total / S, alpha > 2 / 3)

def Smax : Rat := 24 * eps0
def Sl : Rat := 4 / 1000

theorem rigidity_constants :
    (RIG_L < 21283 / 10000 && RIG_M < 949 / 100 &&
     match rigidity Sl with
     | some E => E ≤ 238 / 100 * Sl * (1 + Sl) && E + Sl ≤ 14 / 1000
     | none => false) = true := by native_decide

theorem bracket_at_Smax :
    (match bracket Smax with | some (b, al) => b > 0 && al | none => false) = true := by native_decide

theorem bracket_at_Sl :
    (match bracket Sl with | some (b, al) => b ≥ 1 / 10 && al | none => false) = true := by native_decide

/-- The bracket on the grid S = k / 10^5, k = 1, ..., 400, decreasing. -/
def grid : List Rat := (List.range 400).map fun k => ((k + 1 : Nat) : Rat) / 100000

theorem bracket_decreasing_on_grid :
    (let vals := grid.map fun S => (bracket S).map (·.1)
     vals.all Option.isSome &&
     (List.range 399).all fun i => (vals.getD i none).getD 0 ≥ (vals.getD (i + 1) none).getD 0) = true := by
  native_decide

end D4NearContact

-- intervals used by the two branch and bounds
#eval (D4NearContact.atLeastOn D4NearContact.negq D4NearContact.qmin [(-1, 1/2)]).2
#eval (D4NearContact.atLeastOn D4NearContact.fpoly D4NearContact.fmin
        [(-1 + D4NearContact.delta, -1/2 - D4NearContact.delta), (-1/2 + D4NearContact.delta, -D4NearContact.delta),
         (D4NearContact.delta, 1/2 - D4NearContact.delta)]).2
#print axioms D4NearContact.division_exact
#print axioms D4NearContact.minus_q_lower_bound
#print axioms D4NearContact.p2_factorisation
#print axioms D4NearContact.f_lower_bound
#print axioms D4NearContact.minors_within_one
#print axioms D4NearContact.dW_bound
#print axioms D4NearContact.assembly
#print axioms D4NearContact.rigidity_roots
#print axioms D4NearContact.rigidity_constants
#print axioms D4NearContact.bracket_at_Smax
#print axioms D4NearContact.bracket_at_Sl
#print axioms D4NearContact.bracket_decreasing_on_grid

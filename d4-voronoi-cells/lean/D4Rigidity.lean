/-
D4Rigidity.lean

The finite content of the lemma on the spectrum of the rigidity operator
(lem:rigidity-spectrum of the paper; multi_cap/rigidity_spectrum.py), computed from the
integral D_4 roots.

With the 24 roots a_i (entries 0, +-1, squared length 2), the 96 tight pairs
(i, j) with <a_i, a_j> = 1, the integral matrix Lambda' (row (i, j): a_j in the
block of i, a_i in the block of j), and 2P (blocks 2I - a_i a_i^T), put

    4N = 2P Lambda'^T Lambda' 2P,        2K = Lambda' 2P Lambda'^T .

Proved here, by exact integer and rational computation:

  * 4N annihilates x (x - 8)(x - 20)(x - 24)(x - 32), and the ranks of 4N - cI
    over the rationals are 66, 67, 88, 75, 88 for c = 0, 8, 20, 24, 32, so the
    eigenvalues 0, 2, 5, 6, 8 of N have multiplicities 30, 29, 8, 21, 8, and
    the singular values of Lambda on the tangent space are 0, 1, sqrt(5/2),
    sqrt 3, 2 with multiplicities 6, 29, 8, 21, 8 (the 24 radial directions
    account for the rest of the kernel of N);
  * 2K annihilates x (x - 4)(x - 10)(x - 12)(x - 16), so E0 =
    (2K - 4I)(2K - 10I)(2K - 12I)(2K - 16I) / 7680 is the projector onto its
    kernel, and every diagonal entry of 7680 E0 is 2400: the projector onto the
    image of Lambda has every diagonal entry 11/16, and trace 66;
  * one direction displaced, with the rotational part removed, has
    |Lambda tau|_1^2 / |tau|_2^2 = 96/11, so no constant of the argument passes
    sqrt(96/11).
  * for thm:local-uniqueness: every root lies in eight tight pairs, the tight inner
    products of a tangent vector sum to zero (the column sums of Lambda' 2P
    vanish), and 12^2 * 11/16 + 1/4 = 397/4.

That these matrices represent the operator of the paper, and that the
eigenvalue bookkeeping gives the singular values, is the paper's argument;
the computations are settled by native_decide.

No `sorry`, no Mathlib; compiles against a bare Lean 4 toolchain.
-/

namespace D4Rigidity

abbrev Vec := Array Int
abbrev Mat := Array (Array Int)

/-- The 24 roots +-e_i +-e_j, i < j. -/
def roots : Array Vec := Id.run do
  let mut out : Array Vec := #[]
  for i in [0:4] do
    for j in [i+1:4] do
      for si in [1, -1] do
        for sj in [1, -1] do
          out := out.push ((((#[0, 0, 0, 0] : Vec).set! i si).set! j sj))
  return out

def dot (a b : Vec) : Int := (List.range a.size).foldl (fun s k => s + a[k]! * b[k]!) 0

def tight : Array (Nat × Nat) := Id.run do
  let mut out : Array (Nat × Nat) := #[]
  for i in [0:24] do
    for j in [i+1:24] do
      if dot roots[i]! roots[j]! == 1 then out := out.push (i, j)
  return out

def zeros (r c : Nat) : Mat := Array.replicate r (Array.replicate c 0)
def ident (n : Nat) (s : Int) : Mat := (List.range n).toArray.map fun i => (Array.replicate n 0).set! i s

def Mat.add (A B : Mat) : Mat := (A.zip B).map fun (r, s) => (r.zip s).map fun (x, y) => x + y
def Mat.sub (A B : Mat) : Mat := (A.zip B).map fun (r, s) => (r.zip s).map fun (x, y) => x - y
def Mat.transpose (A : Mat) : Mat :=
  let c := (A.getD 0 #[]).size
  (List.range c).toArray.map fun j => A.map fun r => r[j]!
def Mat.mul (A B : Mat) : Mat :=
  let Bt := B.transpose
  A.map fun r => Bt.map fun c => (List.range r.size).foldl (fun s k => s + r[k]! * c[k]!) 0
def Mat.isZero (A : Mat) : Bool := A.all fun r => r.all (· == 0)
def Mat.symmetric (A : Mat) : Bool := A == A.transpose

/-- Lambda' on (R^4)^24: row (i, j) holds a_j in the block of i and a_i in the block of j. -/
def lam : Mat := tight.map fun (i, j) => Id.run do
  let mut row : Vec := Array.replicate 96 0
  for k in [0:4] do
    row := row.set! (4 * i + k) (row[4 * i + k]! + roots[j]![k]!)
    row := row.set! (4 * j + k) (row[4 * j + k]! + roots[i]![k]!)
  return row

/-- 2P: blocks 2I - a_i a_i^T. -/
def p2 : Mat := (List.range 96).toArray.map fun r =>
  (List.range 96).toArray.map fun c =>
    if r / 4 != c / 4 then 0
    else (if r == c then 2 else 0) - roots[r / 4]![r % 4]! * roots[r / 4]![c % 4]!

def n4 : Mat := p2.mul ((lam.transpose.mul lam).mul p2)
def k2 : Mat := lam.mul (p2.mul lam.transpose)

def shift (A : Mat) (c : Int) : Mat := A.sub (ident A.size c)

def toRat (x : Int) : Rat := x

/-- Rank over the rationals, by Gaussian elimination. -/
def rank (A : Mat) : Nat := Id.run do
  let mut m : Array (Array Rat) := A.map (·.map toRat)
  let rows := m.size
  let cols := (m.getD 0 #[]).size
  let mut r := 0
  for c in [0:cols] do
    if r < rows then
      let mut piv : Option Nat := none
      for k in [r:rows] do
        if piv.isNone && m[k]![c]! != 0 then piv := some k
      match piv with
      | none => pure ()
      | some k =>
        let rk := m[k]!
        m := (m.set! k m[r]!).set! r rk
        let prow := m[r]!
        let pv := prow[c]!
        for k2 in [r+1:rows] do
          let f := m[k2]![c]! / pv
          if f != 0 then
            m := m.set! k2 ((m[k2]!.zip prow).map fun (x, y) => x - f * y)
        r := r + 1
  return r

theorem sizes : (roots.size == 24 && tight.size == 96 && lam.size == 96 && n4.symmetric) = true := by
  native_decide

theorem n4_annihilated :
    (n4.mul ((shift n4 8).mul ((shift n4 20).mul ((shift n4 24).mul (shift n4 32))))).isZero = true := by
  native_decide

theorem n4_ranks :
    ([0, 8, 20, 24, 32].map fun c => rank (shift n4 c)) = [66, 67, 88, 75, 88] := by
  native_decide

/-- The multiplicities 96 - rank: 30, 29, 8, 21, 8, summing to 96. -/
theorem multiplicities :
    (([0, 8, 20, 24, 32].map fun c => 96 - rank (shift n4 c)) = [30, 29, 8, 21, 8] &&
     [30, 29, 8, 21, 8].foldl (· + ·) 0 == 96) = true := by
  native_decide

def e0x7680 : Mat := (shift k2 4).mul ((shift k2 10).mul ((shift k2 12).mul (shift k2 16)))

theorem k2_annihilated : (k2.mul e0x7680).isZero = true := by native_decide

/-- Every diagonal entry of 7680 E0 is 2400 = 7680 * 5/16, so the projector I - E0 onto
the image of Lambda has diagonal 11/16; its trace is 96 * 11/16 = 66. -/
theorem image_diagonal : ((List.range 96).all fun p => e0x7680[p]![p]! == 2400) = true := by
  native_decide

/-! ### One direction displaced -/

/-- The six infinitesimal rotations, as vectors of (R^4)^24: x_i = M_ab a_i. -/
def rotations : Array (Array Rat) := Id.run do
  let mut out : Array (Array Rat) := #[]
  for a in [0:4] do
    for b in [a+1:4] do
      let v : Array Rat := (List.range 96).toArray.map fun idx =>
        let i := idx / 4; let k := idx % 4
        -- (M_ab x)_k = x_b if k = a, -x_a if k = b
        if k == a then (roots[i]![b]! : Rat) else if k == b then -(roots[i]![a]! : Rat) else 0
      out := out.push v
  return out

def rdot (a b : Array Rat) : Rat := (List.range a.size).foldl (fun s k => s + a[k]! * b[k]!) 0

/-- 2P e_2 at root 0, with its components along the (mutually orthogonal) rotations removed. -/
def tau : Array Rat := Id.run do
  let mut x : Array Rat := Array.replicate 96 0
  for k in [0:4] do
    x := x.set! k ((p2[k]![2]! : Rat))
  for r in rotations do
    let c := rdot x r / rdot r r
    x := (x.zip r).map fun (u, v) => u - c * v
  return x

def rabs (x : Rat) : Rat := if x < 0 then -x else x

theorem rotations_orthogonal :
    ((List.range 6).all fun i => (List.range 6).all fun j =>
      i == j || rdot rotations[i]! rotations[j]! == 0) = true := by native_decide

/-- |Lambda tau|_1^2 / |tau|_2^2 = 96/11, with Lambda = Lambda' / sqrt 2. -/
theorem one_direction :
    (let lt : Array Rat := lam.map fun row => rdot (row.map toRat) tau
     (lt.foldl (fun s v => s + rabs v) 0) ^ 2 / (2 * rdot tau tau) == 96 / 11) = true := by
  native_decide

/-! ### The arithmetic of the radius 2/sqrt 397 (thm:local-uniqueness) -/

/-- Every root lies in exactly eight tight pairs, and the tight inner products of a
tangent vector sum to zero: the column sums of Lambda' 2P vanish (the constant stress,
restricted to the tangent space). -/
theorem stress_on_tangent :
    ((List.range 24).all (fun i => (tight.filter fun (a, b) => a == i || b == i).size == 8) &&
     (Mat.mul #[Array.replicate 96 (1 : Int)] (Mat.mul lam p2)).isZero) = true := by
  native_decide

/-- (12 * sqrt 11 / 4)^2 = 99 and 99 + 1/4 = 397/4: from ||Lambda tau||_1 <= 12 ||eps||^2,
||tau||_2 <= (sqrt 11/4) ||Lambda tau||_1 and sum rho_i^2 <= ||eps||^4 / 4. -/
theorem radius_arithmetic : ((12 : Rat) ^ 2 * 11 / 16 == 99 && (99 : Rat) + 1 / 4 == 397 / 4) = true := by
  native_decide

end D4Rigidity

#print axioms D4Rigidity.sizes
#print axioms D4Rigidity.n4_annihilated
#print axioms D4Rigidity.n4_ranks
#print axioms D4Rigidity.multiplicities
#print axioms D4Rigidity.k2_annihilated
#print axioms D4Rigidity.image_diagonal
#print axioms D4Rigidity.rotations_orthogonal
#print axioms D4Rigidity.one_direction
#print axioms D4Rigidity.stress_on_tangent
#print axioms D4Rigidity.radius_arithmetic

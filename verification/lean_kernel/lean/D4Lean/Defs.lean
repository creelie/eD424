import Mathlib
import D4Lean.Check

/-!
# The statements of the certificate

A configuration of two lattice translates `Λ ∪ (Λ + b)` in `ℝ⁴` with minimal distance `2` is
recorded by the Gram matrix `J` of `(a₁, …, a₄, b)`, where `a₁, …, a₄` is a basis of `Λ`.
The minimal distance is at least `2` exactly when `J[k] ≥ 4` for every `k` in
`M* = (ℤ⁴ × {0} \ {0}) ∪ (ℤ⁴ × {±1})`. `R` is the set of symmetric `5 × 5` real matrices with
this property, and `Qb J` is the leading `4 × 4` block, the Gram matrix of `Λ`.
-/

open Matrix

namespace D4

/-- Symmetric `5 × 5` real matrices are written in this type. -/
abbrev M5 := Matrix (Fin 5) (Fin 5) ℝ

/-- `M* = (ℤ⁴ × {0} \ {0}) ∪ (ℤ⁴ × {±1})`. -/
def Mstar : Set (Fin 5 → ℤ) := {k | (k 4 = 0 ∧ k ≠ 0) ∨ k 4 = 1 ∨ k 4 = -1}

/-- `J[k] = ∑ᵢ ∑ⱼ kᵢ Jᵢⱼ kⱼ`. -/
def qform (J : M5) (k : Fin 5 → ℤ) : ℝ := ∑ i, ∑ j, (k i : ℝ) * J i j * (k j : ℝ)

/-- The set `R`. -/
def Rset : Set M5 := {J | J.IsSymm ∧ ∀ k ∈ Mstar, 4 ≤ qform J k}

/-- The leading `4 × 4` block. -/
def Qb (J : M5) : Matrix (Fin 4) (Fin 4) ℝ := J.submatrix Fin.castSucc Fin.castSucc

/-- The group `Γ` of integral matrices with integral inverse whose last row is `(0, …, 0, ±1)`;
it acts on `R` by `J ↦ Tᵀ J T`. -/
def Gamma : Set (Matrix (Fin 5) (Fin 5) ℤ) :=
  {T | IsUnit T.det ∧ (∀ j : Fin 4, T 4 j.castSucc = 0) ∧ (T 4 4 = 1 ∨ T 4 4 = -1)}

/-- The integral matrix of a list matrix. -/
def toZ (A : Mat) : Matrix (Fin 5) (Fin 5) ℤ := Matrix.of fun i j => ent A i j

/-- The real matrix of a list matrix. -/
def toR (A : Mat) : M5 := (toZ A).map (Int.cast)

/-- What the certificate shows about a vertex `v`: `J ∈ R`; its minimal vectors (the `k ∈ M*`
with `J[k] = 4`) are the listed vectors up to sign; `det J` and `det Q` are the listed values,
`det J > 0` or `det Q ≥ 256`, and `det J = 0 ∧ det Q = 256` exactly for the marked vertex. -/
structure VFacts (v : Vtx) : Prop where
  mem : toR v.J ∈ Rset
  mins : ∀ k ∈ Mstar, qform (toR v.J) k = 4 ↔ (List.ofFn k ∈ v.mins ∨ List.ofFn (-k) ∈ v.mins)
  det : (toR v.J).det = v.dJ
  detQ : (Qb (toR v.J)).det = v.dQ
  bound : 0 < v.dJ ∨ 256 ≤ v.dQ
  eq : v.dJ = 0 ∧ v.dQ = 256 ↔ v.eq = true

/-- The edge `J + t Y` leaving the vertex `v` in the direction of `e`. -/
def seg (v : Vtx) (e : Edge) (t : ℝ) : M5 := toR v.J + t • toR e.Y

/-- What the certificate shows about an edge `e` at the vertex `v` (ending at the vertex `w` if it
is bounded): along the edge, `det J > 0` or `det Q ≥ 256`, and `det J = 0 ∧ det Q = 256` only at the
listed points `u / wd`. A bounded edge ends at `t* = tn / td` at a point `Tᵀ J_w T` with `T ∈ Γ`,
and leaves `R` after `t*`; on an unbounded edge `Y[k] ≥ 0` for every `k ∈ M*`, so `J + tY ∈ R` for
every `t ≥ 0`. -/
structure EFacts (v w : Vtx) (e : Edge) : Prop where
  symm : (toR e.Y).IsSymm
  sign : ∀ t : ℝ, 0 ≤ t → (e.bounded = true → t ≤ (e.tn : ℝ) / e.td) →
    0 < (seg v e t).det ∨ 256 ≤ (Qb (seg v e t)).det
  eq : ∀ t : ℝ, 0 ≤ t → (e.bounded = true → t ≤ (e.tn : ℝ) / e.td) →
    (seg v e t).det = 0 → (Qb (seg v e t)).det = 256 → ∃ u ∈ e.eqs, t = (u : ℝ) / e.wd
  bounded : e.bounded = true →
    0 < (e.tn : ℝ) / e.td ∧
    (∃ T ∈ Gamma, seg v e ((e.tn : ℝ) / e.td) = (T.map Int.cast)ᵀ * toR w.J * T.map Int.cast) ∧
    ∃ k ∈ Mstar, ∀ t : ℝ, (e.tn : ℝ) / e.td < t → qform (seg v e t) k < 4
  unbounded : e.bounded = false → ∀ k ∈ Mstar, 0 ≤ qform (toR e.Y) k

end D4

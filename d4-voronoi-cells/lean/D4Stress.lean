/-
D4Stress.lean

A machine check of the integer arithmetic behind prop:deletion-rigid of
"The Sphere Packing Problem in Dimension 4 and the Twenty-Four-Cell Conjecture": the equilibrium
stress on the eighty-eight tight pairs of a deletion configuration.

Everything here is finite arithmetic over the integers, so every statement
is settled by `decide` and the kernel checks it. There is no `sorry` and no
dependence on Mathlib: this file compiles against a bare Lean 4 toolchain.

The roots are taken unnormalised, with squared length 2, exactly as in the
proof of prop:deletion-rigid, so that every quantity below is an integer. In
that scaling the contact condition reads `dot a b <= 1` and a pair is tight
when `dot a b = 1`.
-/

namespace D4Stress

abbrev Vec := Int × Int × Int × Int

def dot (a b : Vec) : Int :=
  a.1 * b.1 + a.2.1 * b.2.1 + a.2.2.1 * b.2.2.1 + a.2.2.2 * b.2.2.2

def add (a b : Vec) : Vec :=
  (a.1 + b.1, a.2.1 + b.2.1, a.2.2.1 + b.2.2.1, a.2.2.2 + b.2.2.2)

def smul (k : Int) (a : Vec) : Vec :=
  (k * a.1, k * a.2.1, k * a.2.2.1, k * a.2.2.2)

def vzero : Vec := (0, 0, 0, 0)

/-- The twenty-four roots of `D_4`, scaled so that each has squared length 2. -/
def roots : List Vec :=
  [ ( 1,  1,  0,  0), ( 1, -1,  0,  0), (-1,  1,  0,  0), (-1, -1,  0,  0),
    ( 1,  0,  1,  0), ( 1,  0, -1,  0), (-1,  0,  1,  0), (-1,  0, -1,  0),
    ( 1,  0,  0,  1), ( 1,  0,  0, -1), (-1,  0,  0,  1), (-1,  0,  0, -1),
    ( 0,  1,  1,  0), ( 0,  1, -1,  0), ( 0, -1,  1,  0), ( 0, -1, -1,  0),
    ( 0,  1,  0,  1), ( 0,  1,  0, -1), ( 0, -1,  0,  1), ( 0, -1,  0, -1),
    ( 0,  0,  1,  1), ( 0,  0,  1, -1), ( 0,  0, -1,  1), ( 0,  0, -1, -1) ]

/-- The root that is deleted. -/
def a0 : Vec := (1, 1, 0, 0)

/-- The deletion configuration: the other twenty-three roots. -/
def W : List Vec := roots.filter (fun a => a != a0)

/-! ## The root system -/

theorem card_roots : roots.length = 24 := by decide

theorem roots_nodup : roots.Nodup := by decide

/-- Every root has squared length 2. -/
theorem roots_norm : (roots.all fun a => dot a a == 2) = true := by decide

/-- Every inner product between distinct roots is `-2`, `-1`, `0` or `1`.
    Normalised, this is `-1`, `-1/2`, `0`, `1/2`, which is the conclusion of
    the theorem "Twenty-four points" of the paper for the root system
    itself. -/
theorem roots_inner :
    (roots.all fun a => roots.all fun b =>
      (a == b) || (dot a b == -2 || dot a b == -1 || dot a b == 0 || dot a b == 1))
      = true := by decide

/-- The contact condition holds: no inner product between distinct roots
    exceeds 1, that is no two roots are closer than 60 degrees. -/
theorem roots_contact :
    (roots.all fun a => roots.all fun b => (a == b) || decide (dot a b ≤ 1)) = true := by
  decide

/-- Every root has exactly eight roots at 60 degrees from it. -/
theorem roots_degree :
    (roots.all fun a => (roots.filter fun b => dot a b == 1).length == 8) = true := by
  decide

/-! ## The deletion configuration -/

theorem card_W : W.length = 23 := by decide

/-- The tight pairs, listed once each. -/
def tightPairs : List (Vec × Vec) :=
  (W.zipIdx.flatMap fun p =>
    (W.zipIdx.filterMap fun q =>
      if p.2 < q.2 && dot p.1 q.1 == 1 then some (p.1, q.1) else none))

/-- Deleting one root leaves eighty-eight tight pairs. -/
theorem tight_count : tightPairs.length = 88 := by decide

/-- The position of a direction relative to the deleted root. -/
def cval (a : Vec) : Int := dot a a0

/-- The four values of `cval` occur with multiplicities 1, 8, 6, 8. -/
theorem cval_multiplicities :
    ((W.filter fun a => cval a == -2).length = 1)
  ∧ ((W.filter fun a => cval a == -1).length = 8)
  ∧ ((W.filter fun a => cval a ==  0).length = 6)
  ∧ ((W.filter fun a => cval a ==  1).length = 8) := by decide

/-- `cval` takes no other value on `W`. -/
theorem cval_range :
    (W.all fun a => cval a == -2 || cval a == -1 || cval a == 0 || cval a == 1)
      = true := by decide

/-! ## The stress -/

/-- The weight attached to a tight pair, by the type of the pair. -/
def yval (p q : Int) : Int :=
  let u := if p ≤ q then p else q
  let v := if p ≤ q then q else p
  if u == -1 && v == -1 then 1
  else if u == -1 && v == 1 then 1
  else if u == -2 && v == -1 then 2
  else if u == -1 && v == 0 then 2
  else if u == 0 && v == 1 then 2
  else if u == 1 && v == 1 then 3
  else 0

/-- The diagonal weight. -/
def mu (a : Vec) : Int := if cval a == -1 then -6 else -8

/-- The six listed types are the only ones that occur among the tight pairs,
    equivalently every tight pair carries a weight in `{1, 2, 3}`. -/
theorem weights_positive :
    (W.all fun a => W.all fun b =>
      (dot a b != 1) ||
      (yval (cval a) (cval b) == 1 || yval (cval a) (cval b) == 2
        || yval (cval a) (cval b) == 3)) = true := by decide

/-- The left-hand side of the equilibrium relation at a direction `a`. -/
def residual (a : Vec) : Vec :=
  add
    (W.foldl (fun acc b =>
      if dot a b == 1 then add acc (smul (yval (cval a) (cval b)) b) else acc) vzero)
    (smul (mu a) a)

/-- **The equilibrium relation.** For every direction of the deletion
    configuration, the weighted sum of its tight neighbours cancels against
    its own multiple of itself. This is equation eq:stress of the paper, and it
    is what makes every first-order motion hold all eighty-eight pairs at
    equality. -/
theorem equilibrium : (W.all fun a => residual a == vzero) = true := by decide

/-! ## The pair count of prop:pair-budget -/

/-- A graph on twenty-three vertices of maximum degree ten has at most
    one hundred and fifteen edges, which is the ceiling quoted in
    prop:pair-budget. -/
theorem pair_ceiling : (23 * 10) / 2 = 115 := by decide

/-- With the integration stopped at r_23, the deletion configuration is
    three pairs short of the ninety-one that the pairwise estimate would
    need (rem:pair-budget). -/
theorem deletion_shortfall : 91 - tightPairs.length = 3 := by decide

/-- Carried to r_*, the estimate needs sixty-five pairs at sixty degrees
    (prop:pair-budget), and the deletion configuration has twenty-three
    more than that. -/
theorem deletion_surplus : tightPairs.length - 65 = 23 := by decide

/-! ## The root system itself (the corollary after prop:deletion-rigid)

With nothing deleted the tight pairs are the ninety-six edges of the
24-cell, and the constant stress, weight 1 on every tight pair and -4 on
the diagonal, is in equilibrium: the eight roots at sixty degrees from a
root sum to four times that root. -/

/-- The tight pairs of the full root system, listed once each. -/
def tightPairs24 : List (Vec × Vec) :=
  (roots.zipIdx.flatMap fun p =>
    (roots.zipIdx.filterMap fun q =>
      if p.2 < q.2 && dot p.1 q.1 == 1 then some (p.1, q.1) else none))

/-- The root system has ninety-six tight pairs. -/
theorem tight_count24 : tightPairs24.length = 96 := by decide

/-- The left-hand side of the equilibrium relation for the constant stress. -/
def residual24 (a : Vec) : Vec :=
  add
    (roots.foldl (fun acc b => if dot a b == 1 then add acc b else acc) vzero)
    (smul (-4) a)

/-- **The equilibrium relation for the root system.** The sum of the eight
    tight neighbours of every root is four times the root. -/
theorem equilibrium24 : (roots.all fun a => residual24 a == vzero) = true := by decide

end D4Stress

/-! ## Axiom audit

Every theorem above must report

  depends on axioms: []

or, where propositional extensionality enters through `decide`,

  depends on axioms: [propext]

and in no case `sorryAx`.
-/

#print axioms D4Stress.card_roots
#print axioms D4Stress.roots_nodup
#print axioms D4Stress.roots_norm
#print axioms D4Stress.roots_inner
#print axioms D4Stress.roots_contact
#print axioms D4Stress.roots_degree
#print axioms D4Stress.card_W
#print axioms D4Stress.tight_count
#print axioms D4Stress.cval_multiplicities
#print axioms D4Stress.cval_range
#print axioms D4Stress.weights_positive
#print axioms D4Stress.equilibrium
#print axioms D4Stress.pair_ceiling
#print axioms D4Stress.deletion_shortfall
#print axioms D4Stress.deletion_surplus
#print axioms D4Stress.tight_count24
#print axioms D4Stress.equilibrium24

/-
D4Closure.lean

A machine check of exact content of "The Sphere Packing Problem in Dimension 4
and the Twenty-Four-Cell Conjecture" (paper/D4.tex; results are named by
their LaTeX labels): the three identities behind lem:holes (holes),
prop:inversion-hull (the inversion hull) and lem:no-triple (no three centres
cut the same point of B(sqrt(3/2))), the identity and the count used in the
proof of thm:m23 (at most twenty-three centres within sqrt 6), and the finite
combinatorics of cor:root-subsets (contacts that contain most of a root
system).

The identities are proved over every commutative ring, with the ring solver
of `grind`, so they hold verbatim over the reals; in the paper each is
followed by a one-line sign argument that uses only |y|, |z| >= 2 and
|y - z| >= 2.  The combinatorics is integer arithmetic on the unnormalised
roots (squared length 2), settled by kernel computation (`decide +kernel`).
There is no `sorry` and no dependence on Mathlib.
-/

set_option maxRecDepth 20000
set_option maxHeartbeats 4000000

namespace D4Closure

open Lean.Grind

/-! ## The identities -/

section Identities
variable {α : Type} [CommRing α]

/-- lem:holes: `|z - 2w|^2 - 4 = |z|^2 - 4<z,w> + 4(|w|^2 - 1)`, in coordinates.
    For a unit contact direction `w` the last term vanishes, so `|z - 2w| >= 2`
    reads `<z,w> <= |z|^2/4`. -/
theorem holes (z1 z2 z3 z4 w1 w2 w3 w4 : α) :
    (z1 - 2*w1)^2 + (z2 - 2*w2)^2 + (z3 - 2*w3)^2 + (z4 - 2*w4)^2 - 4
      = (z1^2 + z2^2 + z3^2 + z4^2) - 4*(z1*w1 + z2*w2 + z3*w3 + z4*w4)
        + 4*((w1^2 + w2^2 + w3^2 + w4^2) - 1) := by
  grind

/-- prop:inversion-hull, with the denominator `2|y|^2` cleared:
    `|y|^2 |z|^2 - 8<y,z> = (|z|^2 - 4)(|y|^2 - 4) + 4(|y - z|^2 - 4)`.
    Dividing by `2|y|^2` gives `|z|^2/2 - <4y/|y|^2, z>` on the left. -/
theorem inversion_hull (y1 y2 y3 y4 z1 z2 z3 z4 : α) :
    (y1^2 + y2^2 + y3^2 + y4^2) * (z1^2 + z2^2 + z3^2 + z4^2)
      - 8*(y1*z1 + y2*z2 + y3*z3 + y4*z4)
    = ((z1^2 + z2^2 + z3^2 + z4^2) - 4) * ((y1^2 + y2^2 + y3^2 + y4^2) - 4)
      + 4*(((y1 - z1)^2 + (y2 - z2)^2 + (y3 - z3)^2 + (y4 - z4)^2) - 4) := by
  grind

/-- lem:no-triple, first identity, one coordinate at a time (the squared distances
    are sums over the four coordinates of exactly these terms):
    `4 sum_{i<j} (p_i - p_j)^2 = sum_i (4 p_i - S)^2`,  S = p_0 + p_1 + p_2 + p_3,
    that is `sum_{i<j} |p_i - p_j|^2 = 4 sum_i |p_i - g|^2` with g = S/4. -/
theorem four_points (a b c d : α) :
    4*((a - b)^2 + (a - c)^2 + (a - d)^2 + (b - c)^2 + (b - d)^2 + (c - d)^2)
      = (4*a - (a+b+c+d))^2 + (4*b - (a+b+c+d))^2
        + (4*c - (a+b+c+d))^2 + (4*d - (a+b+c+d))^2 := by
  grind

/-- lem:no-triple, second identity, one coordinate at a time:
    `16 sum_i (p_i - x)^2 = sum_i (4 p_i - S)^2 + 4 (S - 4x)^2`,
    that is `sum_i |p_i - x|^2 = sum_i |p_i - g|^2 + 4|g - x|^2`: the centroid
    minimises the sum of squared distances. -/
theorem centroid (a b c d x : α) :
    16*((a - x)^2 + (b - x)^2 + (c - x)^2 + (d - x)^2)
      = (4*a - (a+b+c+d))^2 + (4*b - (a+b+c+d))^2
        + (4*c - (a+b+c+d))^2 + (4*d - (a+b+c+d))^2
        + 4*((a+b+c+d) - 4*x)^2 := by
  grind

/-- thm:m23, range II_s: the packing bound `a(d1, d2) = (d1^2 + d2^2 - 4)/(2 d1 d2)`
    lies below its tangent plane `1/2 + (d1 + d2 - 4)/4` at `(2, 2)`; with the
    denominator `4 d1 d2` cleared the difference is
    `d1 d2 (d1 + d2 - 2) - 2 d1^2 - 2 d2^2 + 8 = (d1 - 2)(d2 - 2)(d1 + d2 + 2)`,
    which is nonnegative for `d1, d2 >= 2`. -/
theorem amax_tangent (d1 d2 : α) :
    d1*d2*(d1 + d2 - 2) - 2*d1^2 - 2*d2^2 + 8 = (d1 - 2)*(d2 - 2)*(d1 + d2 + 2) := by
  grind

end Identities

/-- thm:m23, Step 2: of 23 centres, each pair lies in 21 triples and each
    centre in `C(22, 2) = 231 = 21 * 11` triples, which fixes the weight 1/11. -/
theorem triple_counts : 23 - 2 = 21 ∧ 22 * 21 / 2 = 231 ∧ 231 = 21 * 11 := by
  decide

/-! ## cor:root-subsets: the orthogonal root pairs at the vertices of the 24-cell -/

abbrev Vec := Int × Int × Int × Int

def dot (a b : Vec) : Int :=
  a.1 * b.1 + a.2.1 * b.2.1 + a.2.2.1 * b.2.2.1 + a.2.2.2 * b.2.2.2

def add (a b : Vec) : Vec := (a.1 + b.1, a.2.1 + b.2.1, a.2.2.1 + b.2.2.1, a.2.2.2 + b.2.2.2)

/-- The twenty-four roots of `D_4`, scaled so that each has squared length 2. -/
def roots : List Vec :=
  [ ( 1,  1,  0,  0), ( 1, -1,  0,  0), (-1,  1,  0,  0), (-1, -1,  0,  0),
    ( 1,  0,  1,  0), ( 1,  0, -1,  0), (-1,  0,  1,  0), (-1,  0, -1,  0),
    ( 1,  0,  0,  1), ( 1,  0,  0, -1), (-1,  0,  0,  1), (-1,  0,  0, -1),
    ( 0,  1,  1,  0), ( 0,  1, -1,  0), ( 0, -1,  1,  0), ( 0, -1, -1,  0),
    ( 0,  1,  0,  1), ( 0,  1,  0, -1), ( 0, -1,  0,  1), ( 0, -1,  0, -1),
    ( 0,  0,  1,  1), ( 0,  0,  1, -1), ( 0,  0, -1,  1), ( 0,  0, -1, -1) ]

/-- The twenty-four vertices of the 24-cell `{x : <x, r> <= 2 for every root r}`
    in the same scaling (squared length 4; the paper's normalisation divides by
    sqrt 2). -/
def verts : List Vec :=
  [ ( 2, 0, 0, 0), (-2, 0, 0, 0), ( 0, 2, 0, 0), ( 0,-2, 0, 0),
    ( 0, 0, 2, 0), ( 0, 0,-2, 0), ( 0, 0, 0, 2), ( 0, 0, 0,-2),
    ( 1, 1, 1, 1), ( 1, 1, 1,-1), ( 1, 1,-1, 1), ( 1, 1,-1,-1),
    ( 1,-1, 1, 1), ( 1,-1, 1,-1), ( 1,-1,-1, 1), ( 1,-1,-1,-1),
    (-1, 1, 1, 1), (-1, 1, 1,-1), (-1, 1,-1, 1), (-1, 1,-1,-1),
    (-1,-1, 1, 1), (-1,-1, 1,-1), (-1,-1,-1, 1), (-1,-1,-1,-1) ]

def idx : List Nat := List.range 24

def root (i : Nat) : Vec := roots.getD i (0, 0, 0, 0)

/-- Unordered pairs of indices `i < j`. -/
def pairs : List (Nat × Nat) := idx.flatMap (fun i => (idx.filter (fun j => i < j)).map (fun j => (i, j)))

/-- The orthogonal pairs whose sum is the vertex `v`. -/
def pairsAt (v : Vec) : List (Nat × Nat) :=
  pairs.filter (fun p => dot (root p.1) (root p.2) == 0 && add (root p.1) (root p.2) == v)

/-- Each vertex lies on exactly six facets `<x, r> = 2` and satisfies every
    other facet inequality. -/
theorem verts_are_vertices :
    verts.all (fun v => roots.all (fun r => decide (dot v r ≤ 2))
      && (roots.filter (fun r => dot v r == 2)).length == 6) = true := by
  decide +kernel

/-- Each vertex is the sum of exactly three orthogonal pairs of roots. -/
theorem three_pairs_each : verts.all (fun v => (pairsAt v).length == 3) = true := by
  decide +kernel

/-- Every orthogonal pair of roots sums to a vertex; there are 72 of them. -/
theorem orthogonal_pairs :
    (pairs.filter (fun p => dot (root p.1) (root p.2) == 0)).length = 72
    ∧ (pairs.filter (fun p => dot (root p.1) (root p.2) == 0)).all
        (fun p => verts.contains (add (root p.1) (root p.2))) = true := by
  decide +kernel

/-- A set `D` of deleted root indices kills the vertex `v` when every one of the
    three pairs at `v` meets `D`. -/
def kills (D : List Nat) (v : Vec) : Bool :=
  (pairsAt v).all (fun p => D.contains p.1 || D.contains p.2)

def killsSome (D : List Nat) : Bool := verts.any (kills D)

/-- No single deleted root and no pair of deleted roots leaves a vertex without
    an intact orthogonal pair. -/
theorem no_kill_one_two :
    idx.all (fun i => !killsSome [i]) = true
    ∧ pairs.all (fun p => !killsSome [p.1, p.2]) = true := by
  decide +kernel

def triples : List (Nat × Nat × Nat) :=
  pairs.flatMap (fun p => (idx.filter (fun k => p.2 < k)).map (fun k => (p.1, p.2, k)))

def killingTriples : List (Nat × Nat × Nat) :=
  triples.filter (fun t => killsSome [t.1, t.2.1, t.2.2])

/-- Of the 2024 triples of deleted roots exactly 96 leave some vertex without an
    intact pair, and every one of them is pairwise at 60 degrees (`dot = 1`)
    with a vertex on all three of its facets. -/
theorem killing_triples :
    triples.length = 2024 ∧ killingTriples.length = 96
    ∧ killingTriples.all (fun t =>
        dot (root t.1) (root t.2.1) == 1 && dot (root t.1) (root t.2.2) == 1
        && dot (root t.2.1) (root t.2.2) == 1
        && verts.any (fun v => dot v (root t.1) == 2 && dot v (root t.2.1) == 2
                                && dot v (root t.2.2) == 2)) = true := by
  decide +kernel

end D4Closure

#print axioms D4Closure.holes
#print axioms D4Closure.inversion_hull
#print axioms D4Closure.four_points
#print axioms D4Closure.centroid
#print axioms D4Closure.amax_tangent
#print axioms D4Closure.triple_counts
#print axioms D4Closure.verts_are_vertices
#print axioms D4Closure.three_pairs_each
#print axioms D4Closure.orthogonal_pairs
#print axioms D4Closure.no_kill_one_two
#print axioms D4Closure.killing_triples

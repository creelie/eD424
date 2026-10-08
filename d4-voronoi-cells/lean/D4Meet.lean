/-
D4Meet.lean

A machine check of the combinatorics behind prop:meet22 and prop:meet21 of
"The Sphere Packing Problem in Dimension 4 and the Twenty-Four-Cell Conjecture": how much of a root
system a contact configuration can hold.

The analytic half of those proofs is three inequalities in the coordinates
of a unit vector, and it is written out in the text. The other half is
finite: it concerns the six supports of the roots, the three couples of
complementary index pairs they form, and the triples of roots that are
pairwise at 60 degrees. That half is settled here by kernel computation.

Everything is integer arithmetic on the unnormalised roots, which have
squared length 2, so the contact condition reads `dot a b <= 1` and a pair
is tight, at 60 degrees, when `dot a b = 1`. There is no `sorry` and no
dependence on Mathlib.
-/

set_option maxRecDepth 8000

namespace D4Meet

abbrev Vec := Int × Int × Int × Int

def dot (a b : Vec) : Int :=
  a.1 * b.1 + a.2.1 * b.2.1 + a.2.2.1 * b.2.2.1 + a.2.2.2 * b.2.2.2

def smul (k : Int) (a : Vec) : Vec :=
  (k * a.1, k * a.2.1, k * a.2.2.1, k * a.2.2.2)

/-- The twenty-four roots of `D_4`, scaled so that each has squared length 2. -/
def roots : List Vec :=
  [ ( 1,  1,  0,  0), ( 1, -1,  0,  0), (-1,  1,  0,  0), (-1, -1,  0,  0),
    ( 1,  0,  1,  0), ( 1,  0, -1,  0), (-1,  0,  1,  0), (-1,  0, -1,  0),
    ( 1,  0,  0,  1), ( 1,  0,  0, -1), (-1,  0,  0,  1), (-1,  0,  0, -1),
    ( 0,  1,  1,  0), ( 0,  1, -1,  0), ( 0, -1,  1,  0), ( 0, -1, -1,  0),
    ( 0,  1,  0,  1), ( 0,  1,  0, -1), ( 0, -1,  0,  1), ( 0, -1,  0, -1),
    ( 0,  0,  1,  1), ( 0,  0,  1, -1), ( 0,  0, -1,  1), ( 0,  0, -1, -1) ]

theorem card_roots : roots.length = 24 := by decide

theorem roots_nodup : roots.Nodup := by decide

/-! ## Supports

The support of a root is the index pair carrying its two nonzero
coordinates. The six supports pair off into three couples of complementary
index pairs, and those three couples carry the whole argument.
-/

def coord (a : Vec) (k : Nat) : Int :=
  match k with
  | 0 => a.1
  | 1 => a.2.1
  | 2 => a.2.2.1
  | _ => a.2.2.2

def pairs : List (Nat × Nat) := [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

def support (a : Vec) : Nat × Nat :=
  (pairs.filter fun p => coord a p.1 != 0 && coord a p.2 != 0).headD (0,0)

/-- Every root has exactly two nonzero coordinates, so exactly one of the six
    index pairs is its support. -/
theorem support_wellposed :
    (roots.all fun a =>
      (pairs.filter fun p => coord a p.1 != 0 && coord a p.2 != 0).length == 1)
      = true := by decide

/-- Each of the six supports carries exactly four roots. -/
theorem support_fibres :
    (pairs.all fun p => (roots.filter fun a => support a == p).length == 4)
      = true := by decide

/-- The three couples of complementary index pairs. -/
def couples : List ((Nat × Nat) × (Nat × Nat)) :=
  [((0,1),(2,3)), ((0,2),(1,3)), ((0,3),(1,2))]

/-- The couples use each of the six supports exactly once. -/
theorem couples_cover :
    (pairs.all fun p =>
      (couples.filter fun c => c.1 == p || c.2 == p).length == 1) = true := by
  decide

/-- A support is filled when none of the listed supports is equal to it. -/
def filled (touched : List (Nat × Nat)) (p : Nat × Nat) : Bool :=
  touched.all fun q => q != p

/-- Some couple has both of its supports filled. -/
def hasFilledCouple (touched : List (Nat × Nat)) : Bool :=
  couples.any fun c => filled touched c.1 && filled touched c.2

/-! ## Two removed roots

Two roots touch at most two of the six supports, so one couple survives
whole. This is the finite half of prop:meet22, and since it depends
only on the supports it is a statement about the thirty-six ordered pairs
of supports.
-/

theorem two_supports_leave_a_couple :
    (pairs.all fun p => pairs.all fun q => hasFilledCouple [p, q]) = true := by
  decide

theorem two_roots_leave_a_couple :
    (roots.all fun a => roots.all fun b =>
      hasFilledCouple [support a, support b]) = true := by decide

/-! ## Three removed roots

Three roots can touch three supports, one from each couple. Then no couple
survives, and the three supports either share an index, a star, or are the
three pairs inside a triple of indices, a triangle.
-/

def isStar (p q r : Nat × Nat) : Bool :=
  [0,1,2,3].any fun k =>
    (p.1 == k || p.2 == k) && (q.1 == k || q.2 == k) && (r.1 == k || r.2 == k)

def isTriangle (p q r : Nat × Nat) : Bool :=
  ([0,1,2,3].filter fun k =>
    ([p.1, p.2, q.1, q.2, r.1, r.2] : List Nat).contains k).length == 3

/-- The support triples that leave no couple whole. There are eight of them
    up to order, so `6^3 = 216` ordered triples of supports yield
    `8 * 6 = 48` ordered ones. -/
def badTriples : List ((Nat × Nat) × (Nat × Nat) × (Nat × Nat)) :=
  (pairs.flatMap fun p => pairs.flatMap fun q => pairs.filterMap fun r =>
    if hasFilledCouple [p, q, r] then none else some (p, q, r))

theorem bad_triples_count : badTriples.length = 48 := by decide

/-- Each of them has three distinct supports. -/
theorem bad_triples_distinct :
    (badTriples.all fun t =>
      (t.1 != t.2.1) && (t.1 != t.2.2) && (t.2.1 != t.2.2)) = true := by decide

/-- Each of them is a star or a triangle, and both kinds occur. -/
theorem bad_triples_star_or_triangle :
    (badTriples.all fun t =>
      isStar t.1 t.2.1 t.2.2 || isTriangle t.1 t.2.1 t.2.2) = true := by decide

theorem bad_triples_split :
    ((badTriples.filter fun t => isStar t.1 t.2.1 t.2.2).length = 24)
  ∧ ((badTriples.filter fun t => isTriangle t.1 t.2.1 t.2.2).length = 24) := by
  decide

/-- Since each support carries four roots, the eight support patterns account
    for `8 * 4^3 = 512` of the `2024` triples of roots. -/
theorem bad_root_triple_count : 8 * 4 * 4 * 4 = 512 := by decide

/-! ## Triples at sixty degrees -/

/-- Summed over ordered tight pairs, the number of roots tight with both
    counts each such triple six times, so there are `576 / 6 = 96` of them. -/
theorem tight_triangle_count :
    ((roots.flatMap fun a => roots.filterMap fun b =>
      if dot a b == 1 then
        some ((roots.filter fun c => dot a c == 1 && dot b c == 1).length)
      else none).sum = 576) := by decide

theorem tight_triangle_count_div : 576 / 6 = 96 := by decide

/-- No four roots are pairwise at 60 degrees: for every tight pair, the roots
    tight with both are pairwise not tight. -/
theorem no_tight_quadruple :
    (roots.all fun a => roots.all fun b => (dot a b != 1) ||
      ((roots.filter fun c => dot a c == 1 && dot b c == 1).all fun c =>
        (roots.filter fun d => dot a d == 1 && dot b d == 1).all fun d =>
          (c == d) || (dot c d != 1))) = true := by decide

/-! ## The standard star, and the map that reaches it -/

def starTriple : List Vec := [(1,1,0,0), (1,0,1,0), (1,0,0,1)]

/-- Its three roots are pairwise at 60 degrees and its supports form a star
    that leaves no couple whole. -/
theorem starTriple_tight :
    (dot (1,1,0,0) (1,0,1,0) == 1) && (dot (1,1,0,0) (1,0,0,1) == 1)
      && (dot (1,0,1,0) (1,0,0,1) == 1)
      && !hasFilledCouple [(0,1), (0,2), (0,3)]
      && isStar (0,1) (0,2) (0,3) = true := by decide

def kept21 : List Vec := roots.filter fun a => !starTriple.contains a

theorem kept21_card : kept21.length = 21 := by decide

/-- The twelve roots supported in the last three coordinates all survive, so
    the three supports inside `{2,3,4}` are filled and the polytope bound of
    the proof applies. -/
theorem kept21_lower_block :
    ((roots.filter fun a => a.1 == 0).length = 12)
  ∧ (((roots.filter fun a => a.1 == 0).all fun a => kept21.contains a)
      = true) := by decide

/-- Each support meeting the first coordinate keeps three of its four roots,
    the missing one being the member of the star. -/
theorem kept21_upper_block :
    (([1, 2, 3] : List Nat).all fun m =>
      ((roots.filter fun a => a.1 != 0 && coord a m != 0).length == 4)
        && ((kept21.filter fun a => a.1 != 0 && coord a m != 0).length == 3))
      = true := by decide

/-- Twice the Hadamard map used in the proof, so that it stays integral. -/
def had2 (a : Vec) : Vec :=
  (a.1 + a.2.1 + a.2.2.1 + a.2.2.2,
   a.1 + a.2.1 - a.2.2.1 - a.2.2.2,
   a.1 - a.2.1 + a.2.2.1 - a.2.2.2,
   a.1 - a.2.1 - a.2.2.1 + a.2.2.2)

/-- It permutes the root system. -/
theorem had2_permutes :
    (roots.all fun a => roots.any fun b => had2 a == smul 2 b) = true := by
  decide

/-- It preserves inner products, so it is an isometry after the scaling. -/
theorem had2_isometry :
    (roots.all fun a => roots.all fun b =>
      dot (had2 a) (had2 b) == 4 * dot a b) = true := by decide

/-- It carries the standard triangle to the standard star, up to the sign of
    the last coordinate. -/
theorem had2_triangle_to_star :
    (had2 (1,1,0,0) == smul 2 (1,1,0,0))
  ∧ (had2 (1,0,1,0) == smul 2 (1,0,1,0))
  ∧ (had2 (0,1,1,0) == smul 2 (1,0,0,-1)) := by decide

/-! ## The configuration left at twenty-three roots -/

def deletion : List Vec := roots.filter fun a => a != ((1,1,0,0) : Vec)

/-- A contact configuration meeting a root system in 23 directions is the
    root system with one root removed, and that configuration carries the
    88 tight pairs of prop:deletion-rigid. -/
theorem deletion_tight_pairs :
    (deletion.length = 23)
  ∧ ((deletion.zipIdx.flatMap fun p =>
       deletion.zipIdx.filterMap fun q =>
         if p.2 < q.2 && dot p.1 q.1 == 1 then some (p.1, q.1) else none).length
      = 88) := by decide

end D4Meet

/-! ## Axiom audit

Every theorem above must report

  depends on axioms: []

or, where propositional extensionality enters through `decide`,

  depends on axioms: [propext]

and in no case `sorryAx`.
-/

#print axioms D4Meet.card_roots
#print axioms D4Meet.roots_nodup
#print axioms D4Meet.support_wellposed
#print axioms D4Meet.support_fibres
#print axioms D4Meet.couples_cover
#print axioms D4Meet.two_supports_leave_a_couple
#print axioms D4Meet.two_roots_leave_a_couple
#print axioms D4Meet.bad_triples_count
#print axioms D4Meet.bad_triples_distinct
#print axioms D4Meet.bad_triples_star_or_triangle
#print axioms D4Meet.bad_triples_split
#print axioms D4Meet.bad_root_triple_count
#print axioms D4Meet.tight_triangle_count
#print axioms D4Meet.tight_triangle_count_div
#print axioms D4Meet.no_tight_quadruple
#print axioms D4Meet.starTriple_tight
#print axioms D4Meet.kept21_card
#print axioms D4Meet.kept21_lower_block
#print axioms D4Meet.kept21_upper_block
#print axioms D4Meet.had2_permutes
#print axioms D4Meet.had2_isometry
#print axioms D4Meet.had2_triangle_to_star
#print axioms D4Meet.deletion_tight_pairs

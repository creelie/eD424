/-
D4Cap.lean

The exact content of the extremal cap theorem (Theorem "Extremal cap of the
cross-polytope" of the paper, with the polytope Q and the cross-polytope
enclosure before it; cap_certificate/cap_inequality_certificate.py, steps A, B,
D, E and F), computed from the integral D_4 roots.

  A.  Q = {z : <z, a> <= 1 for the 23 roots a other than a_0}: its vertices,
      enumerated exactly over all 4-subsets of the 23 roots, are 25; the 24-cell
      {<z, a> <= 1 for all 24 roots} has 24; Q has exactly one vertex beyond the
      facet <z, a_0> = 1, so Q is the 24-cell with a pyramid on that facet; and,
      by exact triangulation (each octahedral facet into four tetrahedra), the
      24-cell has volume 8 and Q has volume 25/3 in the coordinates x = sqrt2 z.
  B.  In the frame of four pairwise orthogonal roots, every vertex of Q has
      l^1 norm at most sqrt 2, attained: sum_k |<E_k, v>| <= 2.
  D.  With D = (1/(2t)) d/dt (the derivative in a = t^2), D^4 of
      g = t^2 (2t - 1)^4 is 3 (20 t^2 - 3) / (2 t^5), and D^3 g vanishes at t = 1/2.
  E.  The three ranges of the case |S| = 1: the polynomial identities, the
      sign of each factor on its range (branch and bound in exact rationals on
      a rational interval containing the range), and the integer inequalities
      113^2 > 2 * 79^2, 163^2 > 3 * 93^2, 23^2 > 2 * 11^2.
  F.  The case |S| >= 2: the box certificate, 303 boxes from a grid of 216,
      every corner value of the divided difference at most 1, the largest
      0.99755..., exactly as the script computes it.

The formula for the cap volume (Lemma "cap formula") and the monotonicity of the
divided difference that make F a proof are the paper's; the finite checks are
settled by native_decide.

No `sorry`, no Mathlib; compiles against a bare Lean 4 toolchain.
-/

namespace D4Cap

abbrev Vec := Array Rat

def roots : Array (Array Int) := Id.run do
  let mut out : Array (Array Int) := #[]
  for i in [0:4] do
    for j in [i+1:4] do
      for si in [1, -1] do
        for sj in [1, -1] do
          out := out.push ((((#[0, 0, 0, 0] : Array Int).set! i si).set! j sj))
  return out

def toRat (x : Int) : Rat := x
def rootQ (i : Nat) : Vec := roots[i]!.map toRat
def dot (a b : Vec) : Rat := (List.range 4).foldl (fun s k => s + a[k]! * b[k]!) 0
def rabs (x : Rat) : Rat := if x < 0 then -x else x

/-- 4 x 4 determinant by cofactor expansion. -/
def det3 (m : Array Vec) : Rat :=
  m[0]![0]! * (m[1]![1]! * m[2]![2]! - m[1]![2]! * m[2]![1]!)
  - m[0]![1]! * (m[1]![0]! * m[2]![2]! - m[1]![2]! * m[2]![0]!)
  + m[0]![2]! * (m[1]![0]! * m[2]![1]! - m[1]![1]! * m[2]![0]!)

def minor (m : Array Vec) (r c : Nat) : Array Vec :=
  ((List.range 4).filter (· != r)).toArray.map fun i =>
    ((List.range 4).filter (· != c)).toArray.map fun j => m[i]![j]!

def det4 (m : Array Vec) : Rat :=
  (List.range 4).foldl (fun s c =>
    s + (if c % 2 == 0 then 1 else -1) * m[0]![c]! * det3 (minor m 0 c)) 0

/-- Solve M z = (1,1,1,1) by Cramer's rule, if det M != 0. -/
def solveOnes (m : Array Vec) : Option Vec :=
  let d := det4 m
  if d == 0 then none else
  some ((List.range 4).toArray.map fun c =>
    det4 (m.map fun row => row.set! c 1) / d)

/-- All 4-subsets of a list of indices. -/
def subsets4 (n : Nat) : List (List Nat) :=
  (List.range n).flatMap fun a => ((List.range n).filter (· > a)).flatMap fun b =>
    ((List.range n).filter (· > b)).flatMap fun c => ((List.range n).filter (· > c)).map fun d => [a, b, c, d]

/-- The vertices of {z : <z, a> <= 1 for a in A}. -/
def vertices (A : Array Vec) : Array Vec := Id.run do
  let mut out : Array Vec := #[]
  for s in subsets4 A.size do
    match solveOnes (s.toArray.map fun i => A[i]!) with
    | none => pure ()
    | some z =>
      if A.all (fun a => dot a z ≤ 1) && !out.contains z then out := out.push z
  return out

def allRoots : Array Vec := (List.range 24).toArray.map rootQ
def a0 : Vec := rootQ 0
def others : Array Vec := ((List.range 24).filter (· != 0)).toArray.map rootQ

def cellVerts : Array Vec := vertices allRoots
def qVerts : Array Vec := vertices others

def sub (a b : Vec) : Vec := (a.zip b).map fun (x, y) => x - y
def dist2 (a b : Vec) : Rat := let d := sub a b; dot d d

/-- Volume (in z) of the pyramid with apex p over an octahedron given by its six vertices:
four tetrahedra about the axis through one pair of opposite vertices. -/
def pyramidOverOctahedron (p : Vec) (oct : Array Vec) : Option Rat := do
  let v0 ← oct[0]?
  -- the vertex opposite v0 is the farthest one
  let far := oct.foldl (fun best w => if dist2 v0 w > dist2 v0 best then w else best) v0
  let ring := oct.filter fun w => w != v0 && w != far
  let r0 ← ring[0]?
  let r0opp := ring.foldl (fun best w => if dist2 r0 w > dist2 r0 best then w else best) r0
  let rest := ring.filter fun w => w != r0 && w != r0opp
  let r1 ← rest[0]?
  let r1opp ← rest[1]?
  if oct.size != 6 || ring.size != 4 || rest.size != 2 then none else
  let cyc := #[r0, r1, r0opp, r1opp]
  let simplexVol (a b c d : Vec) : Rat :=
    rabs (det4 #[sub a p, sub b p, sub c p, sub d p]) / 24
  return (List.range 4).foldl (fun s i =>
    s + simplexVol v0 far cyc[i]! cyc[(i + 1) % 4]!) 0

def origin : Vec := #[0, 0, 0, 0]

def facetOf (V : Array Vec) (a : Vec) : Array Vec := V.filter fun v => dot a v == 1

/-- Volume of the 24-cell in z: the pyramids from the origin over its 24 facets. -/
def cellVolZ : Option Rat :=
  allRoots.foldlM (fun s a => do
    let v ← pyramidOverOctahedron origin (facetOf cellVerts a)
    return s + v) 0

/-- The vertices of Q beyond the facet <z, a_0> = 1. -/
def beyond : Array Vec := qVerts.filter fun v => dot a0 v > 1

def pyramidVolZ : Option Rat := do
  let apex ← beyond[0]?
  pyramidOverOctahedron apex (facetOf cellVerts a0)

theorem vertex_counts :
    (cellVerts.size == 24 && qVerts.size == 25 && beyond.size == 1 &&
     cellVerts.all (fun v => qVerts.contains v)) = true := by native_decide

/-- In x = sqrt2 z volumes scale by 4: the 24-cell has volume 8 and Q has 25/3. -/
theorem volumes :
    (cellVolZ.map (4 * ·) == some 8 &&
     (do let c ← cellVolZ; let p ← pyramidVolZ; return 4 * (c + p)) == some (25 / 3)) = true := by
  native_decide

/-! ### B. The cross-polytope enclosure -/

def frame : Array Vec := Id.run do
  let mut fr : Array Vec := #[a0]
  for i in [1:24] do
    let r := rootQ i
    if fr.size < 4 && fr.all (fun f => dot r f == 0) then fr := fr.push r
  return fr

def l1scaled (v : Vec) : Rat := frame.foldl (fun s e => s + rabs (dot e v)) 0

/-- The l^1 norm of a vertex in the frame is l1scaled / sqrt 2, so at most sqrt 2
exactly when l1scaled <= 2. -/
theorem cross_polytope :
    (frame.size == 4 &&
     ((List.range 4).all fun i => (List.range 4).all fun j => i == j || dot frame[i]! frame[j]! == 0) &&
     qVerts.all (fun v => l1scaled v ≤ 2) && qVerts.any (fun v => l1scaled v == 2)) = true := by
  native_decide

/-! ### Polynomials over the rationals -/

abbrev Poly := List Rat

def eval (p : Poly) (x : Rat) : Rat := p.foldr (fun c acc => c + x * acc) 0
def trim (p : Poly) : Poly := (p.reverse.dropWhile (· == 0)).reverse
def padd (p q : Poly) : Poly :=
  (List.range (Nat.max p.length q.length)).map fun k => p.getD k 0 + q.getD k 0
def pscale (a : Rat) (p : Poly) : Poly := p.map (a * ·)
def pmul (p q : Poly) : Poly :=
  p.zipIdx.foldl (fun acc (a, i) => padd acc (List.replicate i 0 ++ q.map (a * ·))) []
def ppow (p : Poly) : Nat → Poly
  | 0 => [1]
  | n + 1 => pmul p (ppow p n)
def deriv (p : Poly) : Poly :=
  match p with
  | [] => []
  | _ :: cs => (List.range cs.length).zipWith (fun i c => ((i : Nat) + 1 : Rat) * c) cs
def peq (p q : Poly) : Bool := trim p == trim q

def X : Poly := [0, 1]
def C (a : Rat) : Poly := [a]

/-! ### D. The fourth derivative of g in a = t^2 -/

/-- A rational function p(t) / t^k, and D = (1/(2t)) d/dt on it:
D(p / t^k) = (t p' - k p) / (2 t^(k+2)). -/
def Dop (f : Poly × Nat) : Poly × Nat :=
  let (p, k) := f
  (pscale (1 / 2) (padd (pmul X (deriv p)) (pscale (-(k : Rat)) p)), k + 2)

def gpoly : Poly := pmul (pmul X X) (ppow [-1, 2] 4)       -- t^2 (2t - 1)^4

/-- D^4 g = p / t^8 with p = t^3 * 3 (20 t^2 - 3) / 2, that is 3 (20 t^2 - 3) / (2 t^5);
and D^3 g = p3 / t^6 vanishes at t = 1/2. -/
theorem fourth_derivative :
    (let d4 := Dop (Dop (Dop (Dop (gpoly, 0))))
     let d3 := Dop (Dop (Dop (gpoly, 0)))
     d4.2 == 8 && peq d4.1 (pmul (ppow X 3) (pscale (3 / 2) [-3, 0, 20])) &&
     eval d3.1 (1 / 2) == 0) = true := by
  native_decide

/-! ### E. The case |S| = 1 -/

def rabsR := rabs
def rmax (x y : Rat) : Rat := if x ≤ y then y else x
def rmin (x y : Rat) : Rat := if x ≤ y then x else y

structure RI where
  lo : Rat
  hi : Rat
def RI.add (x y : RI) : RI := ⟨x.lo + y.lo, x.hi + y.hi⟩
def RI.mul (x y : RI) : RI :=
  let a := x.lo * y.lo; let b := x.lo * y.hi; let c := x.hi * y.lo; let d := x.hi * y.hi
  ⟨rmin (rmin a b) (rmin c d), rmax (rmax a b) (rmax c d)⟩
def evalIv (p : Poly) (x : RI) : RI := p.foldr (fun c acc => RI.add ⟨c, c⟩ (RI.mul x acc)) ⟨0, 0⟩

/-- p > 0 on [a, b]: every piece of a bisection has a positive lower bound from interval
evaluation; (status, pieces), status 0 when shown. -/
def positiveOn (p : Poly) (a b : Rat) : Nat × Nat :=
  let rec go (fuel done : Nat) (stack : List (Rat × Rat)) : Nat × Nat :=
    match fuel, stack with
    | 0, _ => (2, done)
    | _, [] => (0, done)
    | fuel + 1, (l, h) :: rest =>
      if (evalIv p ⟨l, h⟩).lo > 0 then go fuel (done + 1) rest
      else if h - l < 1 / 2 ^ 30 then (1, done)
      else let m := (l + h) / 2; go fuel (done + 1) ((l, m) :: (m, h) :: rest)
  go 100000 0 [(a, b)]

-- rational brackets: 7071/10000 < 1/sqrt2 < 7072/10000, 866/1000 < sqrt3/2 < 8661/10000
theorem brackets :
    ((7071 / 10000 : Rat) ^ 2 < 1 / 2 && (1 / 2 : Rat) < (7072 / 10000) ^ 2 &&
     (866 / 1000 : Rat) ^ 2 < 3 / 4 && (3 / 4 : Rat) < (8661 / 10000) ^ 2) = true := by native_decide

def cubic1 : Poly := [-1, 7, -18, 14]                 -- 14c^3 - 18c^2 + 7c - 1
def q2 : Poly := [13, -102, 200, -112]                -- -112c^3 + 200c^2 - 102c + 13
def r3 : Poly := [1, 6, 44, -56]                      -- -56c^3 + 44c^2 + 6c + 1

/-- c^2 (2c^2 - 1) - (2c - 1)^4 = (1 - c)(14c^3 - 18c^2 + 7c - 1);
(c^2 - 1/4)(2c^2 - 3/4) - (2c - 1)^4 = (1/16)(2c - 1) q(c);
8((c + 1/2)^3 - 8(c - 1/2)c^2) = r(c). -/
theorem case_identities :
    (peq (padd (pmul (pmul X X) [-1, 0, 2]) (pscale (-1) (ppow [-1, 2] 4))) (pmul [1, -1] cubic1) &&
     peq (padd (pmul [-1/4, 0, 1] [-3/4, 0, 2]) (pscale (-1) (ppow [-1, 2] 4)))
         (pscale (1 / 16) (pmul [-1, 2] q2)) &&
     peq (pscale 8 (padd (ppow [1/2, 1] 3) (pscale (-8) (pmul [-1/2, 1] (pmul X X))))) r3) = true := by
  native_decide

/-- The cubic is positive on [0.866, 1] (so on [sqrt3/2, 1]), q on [0.7071, 0.8661]
(so on [1/sqrt2, sqrt3/2]), r on [1/2, 0.7072] (so on [1/2, 1/sqrt2]). -/
theorem case_signs :
    ((positiveOn cubic1 (866 / 1000) 1).1 == 0 &&
     (positiveOn q2 (7071 / 10000) (8661 / 10000)).1 == 0 &&
     (positiveOn r3 (1 / 2) (7072 / 10000)).1 == 0) = true := by native_decide

theorem endpoint_integers :
    ((113 : Int) ^ 2 > 2 * 79 ^ 2 && (163 : Int) ^ 2 > 3 * 93 ^ 2 && (23 : Int) ^ 2 > 2 * 11 ^ 2) = true := by
  native_decide

/-! ### F. The case |S| >= 2: the box certificate -/

def gAt (t : Rat) : Rat := if t ≤ 1 / 2 then 0 else t * t * (2 * t - 1) ^ 4

/-- g[a_0, ..., a_3] at a_k = t_k^2 (nodes distinct). -/
def divDiff (ts : List Rat) : Rat := Id.run do
  let a := ts.toArray.map fun x => x * x
  let mut v := ts.toArray.map gAt
  for k in [1:4] do
    v := (List.range (4 - k)).toArray.map fun i => (v[i + 1]! - v[i]!) / (a[i + k]! - a[i]!)
  return v[0]!

def EPS : Rat := 1 / 10 ^ 6
def DEN : Nat := 10 ^ 6

def insertDesc (x : Rat) : List Rat → List Rat
  | [] => [x]
  | y :: ys => if x ≥ y then x :: y :: ys else y :: insertDesc x ys
def sortDesc (l : List Rat) : List Rat := l.foldl (fun acc x => insertDesc x acc) []

/-- The upper bound of the script: nodes at the upper corner (the fourth from the sum),
separated by EPS, lifted to rationals t_k with t_k^2 above them, ties broken downward
from the top. -/
def cornerBound (lo hi : Array Rat) : Rat :=
  let a3 := rmax 0 (1 - lo[0]! - lo[1]! - lo[2]!)
  let highs := sortDesc [hi[0]!, hi[1]!, hi[2]!, a3]
  let highs := highs.zipIdx.map fun (x, i) => x + (i : Rat) * EPS
  let ts := highs.map fun x => ((Nat.sqrt (x * ((DEN * DEN : Nat) : Rat)).floor.toNat + 1 : Nat) : Rat) / (DEN : Rat)
  let ts := sortDesc ts
  let ts := (List.range 4).foldl (fun acc i =>
    if i == 0 then acc else
    let prev := acc.getD (i - 1) 0
    if acc.getD i 0 ≥ prev then acc.set i (prev - 1 / 10 ^ 7) else acc) ts
  divDiff ts

def feasible (lo hi : Array Rat) : Bool :=
  !(lo[0]! + lo[1]! + lo[2]! > 1) && !(hi[0]! + hi[1]! + hi[2]! + hi[2]! < 1) &&
  !(hi[1]! < 1 / 4) && !(hi[0]! < lo[1]! || hi[1]! < lo[2]!)

/-- (status, boxes, largest corner value), as the script: depth-first, the last box first. -/
def boxCertificate : Nat × Nat × Rat := Id.run do
  let step : Rat := 3 / (4 * 6)
  let mut stack : List (Array Rat × Array Rat × Nat) := []
  for i in [0:6] do
    for j in [0:6] do
      for k in [0:6] do
        stack := stack ++ [(#[i * step, j * step, k * step], #[(i + 1) * step, (j + 1) * step, (k + 1) * step], 0)]
  let mut nBoxes := 0
  let mut largest : Rat := 0
  let mut fuel := 1000000
  while fuel > 0 do
    fuel := fuel - 1
    match stack.getLast? with
    | none => return (0, nBoxes, largest)
    | some (lo, hi, depth) =>
      stack := stack.dropLast
      if feasible lo hi then
        let value := cornerBound lo hi
        nBoxes := nBoxes + 1
        if value ≤ 1 then
          if value > largest then largest := value
        else
          if depth ≥ 25 then return (1, nBoxes, largest)
          let widths := (List.range 3).map fun m => hi[m]! - lo[m]!
          let m := (List.range 3).foldl (fun b i => if widths.getD i 0 > widths.getD b 0 then i else b) 0
          let mid := (lo[m]! + hi[m]!) / 2
          stack := stack ++ [(lo, hi.set! m mid, depth + 1), (lo.set! m mid, hi, depth + 1)]
  return (2, nBoxes, largest)

theorem box_certificate :
    (boxCertificate.1 == 0 && boxCertificate.2.1 == 303 &&
     boxCertificate.2.2 ≤ 1 && boxCertificate.2.2 > 9975 / 10000) = true := by native_decide

end D4Cap

#eval (D4Cap.boxCertificate.2.1, (D4Cap.boxCertificate.2.2.num.toFloat / D4Cap.boxCertificate.2.2.den.toFloat))
#eval ((D4Cap.positiveOn D4Cap.cubic1 (866 / 1000) 1).2, (D4Cap.positiveOn D4Cap.q2 (7071 / 10000) (8661 / 10000)).2,
       (D4Cap.positiveOn D4Cap.r3 (1 / 2) (7072 / 10000)).2)
#print axioms D4Cap.vertex_counts
#print axioms D4Cap.volumes
#print axioms D4Cap.cross_polytope
#print axioms D4Cap.fourth_derivative
#print axioms D4Cap.brackets
#print axioms D4Cap.case_identities
#print axioms D4Cap.case_signs
#print axioms D4Cap.endpoint_integers
#print axioms D4Cap.box_certificate

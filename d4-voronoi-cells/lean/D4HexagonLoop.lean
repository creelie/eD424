/-
D4HexagonLoop.lean: the algebra of prop:hexagon-loop of the paper, a closed
curve of direction sets through the root system along which the contact cell
{x : <x, w_i> <= 1} has volume exactly 8.

Coordinates R^4 = P + P', e(a) = (cos a, sin a).  At theta = 0 the family is
the unit D_4 root system: the six (e(60m), 0) and the eighteen
(s e(a), c e(b)), (a, b) in {30,150,270} x {0,120,240} or {90,210,330} x
{60,180,300} (degrees), s^2 = 1/3.  Along the curve the six are turned to
(e(theta + 60m), 0) and s = 4C/(4C^2 + 3), C = cos(pi/6 - theta).

The paper writes the volume as an integral over the hexagon Hex_theta of the
areas of hexagonal slices, and reduces it to two integrals J1, J2 over one
sector (two triangles, with vertices O, a30/C, the vertex V of Hex_theta, and
a90/C), times six.  Proved here, by exact computation in the ring
Q(sqrt3)[c, s]/(c^2 + s^2 - 1), c = cos theta, s = sin theta (every element
is A(c) + s B(c) uniquely, so an element is zero exactly when A = B = 0):

  * gram_at_zero: at theta = 0 the Gram matrix of the 24 directions equals
    that of the unit roots (+-1, +-1, 0, 0)/sqrt2 under an explicit labelling;
  * closed_forms: J1 C^2 = sqrt3 C (4C^2 + 3)/3 and
    J2 C^3 = sqrt3 C (16C^4 - 8C^2 + 9)/12, from the triangle integrals;
  * key_identity: 3 J1^2 - 4 sqrt3 J2 - 32 = 0 (times C^4);
  * edge_conditions: the conditions for no slice to degenerate, at the
    vertices of the two triangles, times 4C^2 + 3, are 4C^2 - 3, 4C^2 + 3 and
    3 -+ 4 sqrt3 C S with S = sin(pi/6 - theta) (and C^2 + S^2 = 1);
  * perfect_square (in any commutative ring): the numerator of vol - 8 is
    (4C - s (4C^2 + 3))^2 / (4C^2).

The slice formula, the reduction to one sector and the inequalities
4C^2 >= 3 and 4 sqrt3 |C S| = 2 sqrt3 |sin 2 phi| <= 3 on the curve are the
paper's argument; hexagon_loop.py checks the same identities with sympy and
the volume numerically (qhull, and 50 digits).

No Mathlib; the computations are settled by native_decide, the identity by
grind.
-/

set_option maxRecDepth 100000
set_option maxHeartbeats 4000000

namespace D4HexagonLoop

open Lean.Grind

section Identities
variable {α : Type} [CommRing α]

/-- With J1 C^2 = sqrt3 C (4C^2+3)/3 and J2 C^2 = sqrt3 (16C^4 - 8C^2 + 9)/12, the product
`4C^2 (12 - 2 sqrt3 s J1 + sqrt3 s^2 J2 - 8 (1 - s^2))` is the left side below. -/
theorem perfect_square (C s : α) :
    48*C^2 - 8*s*C*(4*C^2 + 3) + s^2*(16*C^4 - 8*C^2 + 9) - 32*C^2*(1 - s^2)
      = (4*C - s*(4*C^2 + 3))^2 := by
  grind

end Identities

/-! ## Q(sqrt 3) and the ring Q(sqrt3)[c, s]/(c^2 + s^2 - 1) -/

/-- `a + b sqrt3`. -/
structure Q3 where
  a : Rat
  b : Rat
deriving BEq, Inhabited

namespace Q3
def ofRat (x : Rat) : Q3 := ⟨x, 0⟩
def sqrt3 : Q3 := ⟨0, 1⟩
instance : Add Q3 := ⟨fun x y => ⟨x.a + y.a, x.b + y.b⟩⟩
instance : Neg Q3 := ⟨fun x => ⟨-x.a, -x.b⟩⟩
instance : Sub Q3 := ⟨fun x y => ⟨x.a - y.a, x.b - y.b⟩⟩
instance : Mul Q3 := ⟨fun x y => ⟨x.a * y.a + 3 * x.b * y.b, x.a * y.b + x.b * y.a⟩⟩
def isZero (x : Q3) : Bool := x.a == 0 && x.b == 0
end Q3

/-- Polynomials in c, coefficient of c^k at index k. -/
abbrev Poly := List Q3

def padd : Poly → Poly → Poly
  | [], q => q
  | p, [] => p
  | x :: p, y :: q => (x + y) :: padd p q

def pscale (k : Q3) (p : Poly) : Poly := p.map (k * ·)

def pmul : Poly → Poly → Poly
  | [], _ => []
  | x :: p, q => padd (pscale x q) (Q3.ofRat 0 :: pmul p q)

def pIsZero (p : Poly) : Bool := p.all Q3.isZero

/-- `A(c) + s B(c)`, with `s^2 = 1 - c^2`. -/
structure El where
  A : Poly
  B : Poly

namespace El
def const (x : Q3) : El := ⟨[x], []⟩
def ofRat (x : Rat) : El := const (Q3.ofRat x)
def c : El := ⟨[Q3.ofRat 0, Q3.ofRat 1], []⟩
def s : El := ⟨[], [Q3.ofRat 1]⟩
def oneMinusC2 : Poly := [Q3.ofRat 1, Q3.ofRat 0, Q3.ofRat (-1)]
instance : Add El := ⟨fun x y => ⟨padd x.A y.A, padd x.B y.B⟩⟩
instance : Neg El := ⟨fun x => ⟨pscale (Q3.ofRat (-1)) x.A, pscale (Q3.ofRat (-1)) x.B⟩⟩
instance : Sub El := ⟨fun x y => x + -y⟩
instance : Mul El := ⟨fun x y =>
  ⟨padd (pmul x.A y.A) (pmul oneMinusC2 (pmul x.B y.B)), padd (pmul x.A y.B) (pmul x.B y.A)⟩⟩
def isZero (x : El) : Bool := pIsZero x.A && pIsZero x.B
def pow (x : El) : Nat → El
  | 0 => ofRat 1
  | n + 1 => x * pow x n
end El

open El

def r3 : El := const Q3.sqrt3
def half : El := ofRat (1 / 2)

abbrev V2 := El × El
def dot (x y : V2) : El := x.1 * y.1 + x.2 * y.2
def det (x y : V2) : El := x.1 * y.2 - x.2 * y.1
def vadd (x y : V2) : V2 := (x.1 + y.1, x.2 + y.2)

/-! ## One sector of the hexagon -/

def a30 : V2 := (r3 * half, half)
def a90 : V2 := (ofRat 0, ofRat 1)
/-- C = cos(pi/6 - theta) = <e(theta), e(30)>. -/
def C : El := r3 * half * El.c + half * El.s
/-- S = sin(pi/6 - theta). -/
def S : El := half * El.c - r3 * half * El.s
/-- The vertex (2/sqrt3) e(theta + 30) of Hex_theta. -/
def V : V2 := (El.c - ofRat (1 / 3) * r3 * El.s, El.s + ofRat (1 / 3) * r3 * El.c)

/-- J1 C^2: the triangles (O, a30/C, V) and (O, V, a90/C), the linear function
m_+ + m_- = <p, a30 + a90>, times six. -/
def J1C2 : El :=
  let L := vadd a30 a90
  det a30 V * (dot L a30 + C * dot L V) + det V a90 * (C * dot L V + dot L a90)

/-- The integral of L M over the two triangles, times 24 C^3. -/
def Q (L M : V2) : El :=
  det a30 V * (dot L a30 * dot M a30 + C * C * dot L V * dot M V
      + (dot L a30 + C * dot L V) * (dot M a30 + C * dot M V))
  + det V a90 * (dot L a90 * dot M a90 + C * C * dot L V * dot M V
      + (dot L a90 + C * dot L V) * (dot M a90 + C * dot M V))

/-- J2 C^3 = 6 * (4 I(m_+, m_-) - I(m_+, m_+) - I(m_-, m_-)) C^3, with I = Q / (24 C^3). -/
def J2C3 : El := half * half * (ofRat 4 * Q a30 a90 - Q a30 a30 - Q a90 a90)

theorem closed_forms :
    ((J1C2 - r3 * C * (ofRat 4 * C * C + ofRat 3) * ofRat (1 / 3)).isZero &&
     (J2C3 - r3 * C * (ofRat 16 * pow C 4 - ofRat 8 * C * C + ofRat 9) * ofRat (1 / 12)).isZero) = true := by
  native_decide

/-- 3 J1^2 - 4 sqrt3 J2 - 32 = 0, times C^4. -/
theorem key_identity :
    (ofRat 3 * J1C2 * J1C2 - ofRat 4 * r3 * J2C3 * C - ofRat 32 * pow C 4).isZero = true := by
  native_decide

/-- The six edge conditions 1 - sigma (2 m_+ - m_-) >= 0, 1 - sigma (2 m_- - m_+) >= 0 at
a30/C, V, a90/C with sigma = 4C/(4C^2 + 3), multiplied by 4C^2 + 3. -/
theorem edge_conditions :
    let k := ofRat 4 * C * C + ofRat 3
    ((k - ofRat 4 * (ofRat 2 * dot a30 a30 - dot a30 a90) - (ofRat 4 * C * C - ofRat 3)).isZero &&
     (k - ofRat 4 * (ofRat 2 * dot a30 a90 - dot a30 a30) - k).isZero &&
     (k - ofRat 4 * C * (ofRat 2 * dot V a30 - dot V a90) - (ofRat 3 - ofRat 4 * r3 * C * S)).isZero &&
     (k - ofRat 4 * C * (ofRat 2 * dot V a90 - dot V a30) - (ofRat 3 + ofRat 4 * r3 * C * S)).isZero &&
     (k - ofRat 4 * (ofRat 2 * dot a90 a30 - dot a90 a90) - k).isZero &&
     (k - ofRat 4 * (ofRat 2 * dot a90 a90 - dot a90 a30) - (ofRat 4 * C * C - ofRat 3)).isZero &&
     (C * C + S * S - ofRat 1).isZero &&
     (dot (El.c, El.s) a30 - C).isZero) = true := by
  native_decide

/-! ## At theta = 0 the family is the root system -/

/-- cos and sin of 30k degrees. -/
def cos30 (k : Nat) : Q3 :=
  match k % 12 with
  | 0 => ⟨1, 0⟩ | 1 => ⟨0, 1/2⟩ | 2 => ⟨1/2, 0⟩ | 3 => ⟨0, 0⟩
  | 4 => ⟨-1/2, 0⟩ | 5 => ⟨0, -1/2⟩ | 6 => ⟨-1, 0⟩ | 7 => ⟨0, -1/2⟩
  | 8 => ⟨-1/2, 0⟩ | 9 => ⟨0, 0⟩ | 10 => ⟨1/2, 0⟩ | _ => ⟨0, 1/2⟩
def sin30 (k : Nat) : Q3 := cos30 (k + 9)
def dot30 (i j : Nat) : Q3 := cos30 i * cos30 j + sin30 i * sin30 j

/-- The 24 directions at theta = 0, as (kind, a/30, b/30): kind 0 the hexagon, kind 1 the others. -/
def pts : List (Nat × Nat × Nat) :=
  ((List.range 6).map fun m => (0, 2 * m, 0)) ++
  ([1, 5, 9].flatMap fun a => [0, 4, 8].map fun b => (1, a, b)) ++
  ([3, 7, 11].flatMap fun a => [2, 6, 10].map fun b => (1, a, b))

/-- Inner products, with sin psi = 1/sqrt3 = sqrt3/3 and cos^2 psi = 2/3. -/
def ip (x y : Nat × Nat × Nat) : Q3 :=
  match x.1, y.1 with
  | 0, 0 => dot30 x.2.1 y.2.1
  | 0, _ => ⟨0, 1/3⟩ * dot30 x.2.1 y.2.1
  | _, 0 => ⟨0, 1/3⟩ * dot30 x.2.1 y.2.1
  | _, _ => Q3.ofRat (1/3) * dot30 x.2.1 y.2.1 + Q3.ofRat (2/3) * dot30 x.2.2 y.2.2

/-- The roots (+-1, +-1, 0, 0) in the order i < j, signs (+,+), (+,-), (-,+), (-,-). -/
def roots : List (List Int) :=
  (List.range 4).flatMap fun i => (List.range 4).flatMap fun j =>
    if i < j then [1, -1].flatMap fun a => [1, -1].map fun b =>
      (List.range 4).map fun k => if k == i then a else if k == j then b else 0
    else []

def idot (x y : List Int) : Int := (List.zipWith (· * ·) x y).foldl (· + ·) 0

def perm : List Nat := [0, 4, 14, 3, 7, 13, 8, 9, 12, 18, 19, 6, 22, 23, 2, 1, 21, 20, 15, 11, 10, 5, 17, 16]

theorem gram_at_zero :
    (pts.length == 24 && roots.length == 24 &&
     (List.range 24).all (fun i => perm.contains i) &&
     (List.range 24).all fun i => (List.range 24).all fun j =>
       ip pts[i]! pts[j]! == Q3.ofRat ((idot roots[perm[i]!]! roots[perm[j]!]! : Rat) / 2)) = true := by
  native_decide

end D4HexagonLoop

#print axioms D4HexagonLoop.perfect_square
#print axioms D4HexagonLoop.closed_forms
#print axioms D4HexagonLoop.key_identity
#print axioms D4HexagonLoop.edge_conditions
#print axioms D4HexagonLoop.gram_at_zero

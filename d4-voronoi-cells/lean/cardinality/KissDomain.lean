/-
KissDomain.lean: the check of the three-point certificates of thm:kissing-stable
(the kissing number is stable) inside Lean, in exact arithmetic.

For a certificate (f_1, ..., f_d; F_0, ..., F_d) of KissData, this module
establishes what multi_cap/certify_cardinality.py establishes:

  1. every f_k >= 0, and every F_k is symmetric and positive definite
     (exact LDL^T over the rationals, every pivot positive);
  2. B = 1 + f(1) + F(1,1,1), exactly, and 24 (1 - e1) - 552 e2 > B - 1;
  3. (i)  P1(u) = f(u) + 3 F(1,u,u) + 1 <= e1 on [-1, top];
  4. (ii) P2(u,v,w) = F(u,v,w) <= e2 on the ordered admissible domain
          -1 <= u <= v <= w <= top, 1 + 2uvw - u^2 - v^2 - w^2 >= 0,
          and P2 is symmetric in its three variables, so that the ordered
          domain covers the admissible domain in [-1, top]^3;
  5. top >= t, e1 <= 10^-6 and e2 <= e2Dec, the decimals of the paper.

The polynomials G_k, S_k, P1 and P2 are expanded here from the Legendre,
Chebyshev and Gegenbauer recurrences; they are scaled by the least odd
integer that makes every coefficient dyadic, and the branch and bound
(second-order Taylor form about the centre of each box, Hessian bounds by
interval evaluation over the box) runs in exact dyadic arithmetic on
numerators and exponents, with no rounding anywhere.  The polynomial
routines and the dyadic arithmetic are those of lean/certificate/D4CertDomain.
-/
import KissData

/-! ## Rational polynomials in three variables -/

abbrev Key := Nat × Nat × Nat
abbrev RPoly := List (Key × Rat)

def keyLe (a b : Key) : Bool :=
  if a.1 != b.1 then a.1 < b.1
  else if a.2.1 != b.2.1 then a.2.1 < b.2.1
  else a.2.2 ≤ b.2.2

/-- Sort the terms by monomial and merge equal monomials, dropping zeros. -/
def RPoly.normalize (p : RPoly) : RPoly :=
  let s := p.mergeSort (fun x y => keyLe x.1 y.1)
  let rec go (cur : Key × Rat) : List (Key × Rat) → List (Key × Rat)
    | [] => if cur.2 == 0 then [] else [cur]
    | y :: rest =>
      if cur.1 == y.1 then go (cur.1, cur.2 + y.2) rest
      else (if cur.2 == 0 then [] else [cur]) ++ go y rest
  match s with
  | [] => []
  | x :: rest => go x rest

def RPoly.add (p q : RPoly) : RPoly := RPoly.normalize (p ++ q)
def RPoly.scale (p : RPoly) (c : Rat) : RPoly := if c == 0 then [] else p.map fun (k, a) => (k, a * c)
def RPoly.mul (p q : RPoly) : RPoly :=
  RPoly.normalize (p.flatMap fun (k1, a) => q.map fun (k2, b) =>
    ((k1.1 + k2.1, k1.2.1 + k2.2.1, k1.2.2 + k2.2.2), a * b))
def RPoly.pow (p : RPoly) : Nat → RPoly
  | 0 => [((0, 0, 0), 1)]
  | n + 1 => RPoly.mul p (RPoly.pow p n)

def U : RPoly := [((1, 0, 0), 1)]
def V : RPoly := [((0, 1, 0), 1)]
def T : RPoly := [((0, 0, 1), 1)]
def ONE : RPoly := [((0, 0, 0), 1)]

/-- Univariate coefficient list -> polynomial in variable `var` (0, 1, 2). -/
def univariate (cs : List Rat) (var : Nat) : RPoly :=
  RPoly.normalize <| (cs.zipIdx).map fun (c, i) =>
    (if var == 0 then (i, 0, 0) else if var == 1 then (0, i, 0) else (0, 0, i), c)

/-- Legendre polynomials P_0, ..., P_d as coefficient lists. -/
def legendre (d : Nat) : List (List Rat) :=
  let rec go (k : Nat) (pk pkm : List Rat) (acc : List (List Rat)) : Nat → List (List Rat)
    | 0 => acc.reverse
    | n + 1 =>
      -- (k+1) P_{k+1} = (2k+1) x P_k - k P_{k-1}
      let a : List Rat := (0 : Rat) :: pk.map (fun c => (2 * k + 1 : Rat) * c)
      let b : List Rat := pkm ++ List.replicate (a.length - pkm.length) (0 : Rat)
      let nxt := (a.zip b).map fun (x, y) => (x - (k : Rat) * y) / ((k : Rat) + 1)
      go (k + 1) nxt pk (nxt :: acc) n
  if d == 0 then [[1]] else go 1 [0, 1] [1] [[0, 1], [1]] (d - 1)

/-- Chebyshev T_0, ..., T_n. -/
def chebyshev (n : Nat) : List (List Rat) :=
  let rec go (tk tkm : List Rat) (acc : List (List Rat)) : Nat → List (List Rat)
    | 0 => acc.reverse
    | m + 1 =>
      let a : List Rat := (0 : Rat) :: tk.map (fun c => 2 * c)
      let b : List Rat := tkm ++ List.replicate (a.length - tkm.length) (0 : Rat)
      let nxt := (a.zip b).map fun (x, y) => x - y
      go nxt tk (nxt :: acc) m
  if n == 0 then [[1]] else go [0, 1] [1] [[0, 1], [1]] (n - 1)

/-- Gegenbauer polynomials of S^3, G_k = U_k / (k+1). -/
def gegenbauer (d : Nat) : List (List Rat) :=
  let rec go (uk ukm : List Rat) (acc : List (List Rat)) : Nat → List (List Rat)
    | 0 => acc.reverse
    | m + 1 =>
      let a : List Rat := (0 : Rat) :: uk.map (fun c => 2 * c)
      let b : List Rat := ukm ++ List.replicate (a.length - ukm.length) (0 : Rat)
      let nxt := (a.zip b).map fun (x, y) => x - y
      go nxt uk (nxt :: acc) m
  let us := if d == 0 then [[1]] else go [0, 2] [1] [[0, 2], [1]] (d - 1)
  (us.zipIdx).map fun (cs, k) => cs.map fun c => c / ((k : Rat) + 1)

/-- ((1-u^2)(1-v^2))^{k/2} P_k((t-uv)/sqrt((1-u^2)(1-v^2))) as a polynomial. -/
def phiPoly (k : Nat) (LC : List (List Rat)) : RPoly :=
  let x := RPoly.add T (RPoly.scale (RPoly.mul U V) (-1))
  let s2 := RPoly.mul (RPoly.add ONE (RPoly.scale (RPoly.mul U U) (-1)))
                      (RPoly.add ONE (RPoly.scale (RPoly.mul V V) (-1)))
  let a := LC.getD k []
  (List.range (k + 1)).foldl (fun acc m =>
    if (m + k) % 2 == 0 then
      let c := a.getD m 0
      if c == 0 then acc
      else RPoly.add acc (RPoly.scale (RPoly.mul (RPoly.pow x m) (RPoly.pow s2 ((k - m) / 2))) c)
    else acc) []

def permuteKey (perm : Nat × Nat × Nat) (k : Key) : Key :=
  let get (i : Nat) : Nat := if i == 0 then k.1 else if i == 1 then k.2.1 else k.2.2
  (get perm.1, get perm.2.1, get perm.2.2)

def RPoly.permute (p : RPoly) (perm : Nat × Nat × Nat) : RPoly :=
  RPoly.normalize (p.map fun (k, c) => (permuteKey perm k, c))

def perms : List (Nat × Nat × Nat) := [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]

/-- The symmetrised entry S_k(u,v,t)[i,j]. -/
def sEntry (i j : Nat) (phik : RPoly) (TC : List (List Rat)) : RPoly :=
  let y := RPoly.mul (RPoly.mul (univariate (TC.getD i []) 0) (univariate (TC.getD j []) 1)) phik
  RPoly.scale (perms.foldl (fun acc pm => RPoly.add acc (RPoly.permute y pm)) []) (1 / 6)

/-- p(1, u, u) as a polynomial in the first variable. -/
def RPoly.at1uu (p : RPoly) : RPoly :=
  RPoly.normalize (p.map fun (k, c) => ((k.2.1 + k.2.2, 0, 0), c))

/-- Sum of the coefficients: the value at (1, 1, 1). -/
def RPoly.at111 (p : RPoly) : Rat := p.foldl (fun s (_, c) => s + c) 0

/-! ## Dyadic numbers and intervals -/

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
def mulInt (a : Dy) (k : Int) : Dy := ⟨a.n * k, a.e⟩
def zero : Dy := ⟨0, 0⟩
def one : Dy := ⟨1, 0⟩
def isNonneg (a : Dy) : Bool := a.n ≥ 0
/-- The exponent k with den = 2^k, if den is a power of two. -/
def log2Exact : Nat → Nat → Option Nat
  | 0, _ => none
  | _, 1 => some 0
  | fuel + 1, d => if d % 2 == 0 then (log2Exact fuel (d / 2)).map (· + 1) else none
def ofRat (q : Rat) : Option Dy :=
  match log2Exact 400 q.den with
  | some k => some ⟨q.num, k⟩
  | none => none
end Dy

structure Iv where
  lo : Dy
  hi : Dy
deriving Inhabited

namespace Iv
def pt (a : Dy) : Iv := ⟨a, a⟩
def add (x y : Iv) : Iv := ⟨x.lo.add y.lo, x.hi.add y.hi⟩
def sub (x y : Iv) : Iv := ⟨x.lo.sub y.hi, x.hi.sub y.lo⟩
def mul (x y : Iv) : Iv :=
  let p1 := x.lo.mul y.lo; let p2 := x.lo.mul y.hi; let p3 := x.hi.mul y.lo; let p4 := x.hi.mul y.hi
  ⟨Dy.min (Dy.min p1 p2) (Dy.min p3 p4), Dy.max (Dy.max p1 p2) (Dy.max p3 p4)⟩
def scale (x : Iv) (c : Dy) : Iv :=
  if c.isNonneg then ⟨x.lo.mul c, x.hi.mul c⟩ else ⟨x.hi.mul c, x.lo.mul c⟩
def absHi (x : Iv) : Dy := Dy.max x.lo.abs x.hi.abs
end Iv

/-! ## Dyadic polynomials and their evaluation -/

structure Mono where
  a : Nat
  b : Nat
  c : Nat
  co : Dy
deriving Inhabited

abbrev DPoly := Array Mono

def toDPoly (p : RPoly) : Option DPoly :=
  p.foldl (fun acc (k, q) =>
    match acc, Dy.ofRat q with
    | some arr, some d => some (arr.push ⟨k.1, k.2.1, k.2.2, d⟩)
    | _, _ => none) (some #[])

def DPoly.deriv (p : DPoly) (var : Nat) : DPoly :=
  p.foldl (fun acc m =>
    match var with
    | 0 => if m.a == 0 then acc else acc.push ⟨m.a - 1, m.b, m.c, m.co.mulInt m.a⟩
    | 1 => if m.b == 0 then acc else acc.push ⟨m.a, m.b - 1, m.c, m.co.mulInt m.b⟩
    | _ => if m.c == 0 then acc else acc.push ⟨m.a, m.b, m.c - 1, m.co.mulInt m.c⟩) #[]

def DPoly.maxDeg (p : DPoly) : Nat := p.foldl (fun d m => Nat.max d (Nat.max m.a (Nat.max m.b m.c))) 0

/-- Powers x^0, ..., x^n of an interval. -/
def ivPowers (x : Iv) (n : Nat) : Array Iv :=
  (List.range n).foldl (fun acc _ => acc.push (Iv.mul (acc.back!) x)) #[Iv.pt Dy.one]

def dyPowers (x : Dy) (n : Nat) : Array Dy :=
  (List.range n).foldl (fun acc _ => acc.push ((acc.back!).mul x)) #[Dy.one]

/-- Enclosure of p on a box. -/
def DPoly.evalIv (p : DPoly) (pu pv pt : Array Iv) : Iv :=
  p.foldl (fun acc m =>
    Iv.add acc (Iv.scale (Iv.mul (Iv.mul (pu[m.a]!) (pv[m.b]!)) (pt[m.c]!)) m.co)) (Iv.pt Dy.zero)

/-- Exact value of p at a dyadic point. -/
def DPoly.evalPt (p : DPoly) (pu pv pt : Array Dy) : Dy :=
  p.foldl (fun acc m =>
    acc.add ((((pu[m.a]!).mul (pv[m.b]!)).mul (pt[m.c]!)).mul m.co)) Dy.zero

/-- Nested form of a polynomial for evaluation: grouped by the exponent of
u, then of v, then of t, each group sorted; the evaluation multiplies each
group by a single power instead of forming every monomial from scratch. -/
abbrev NPoly := Array (Nat × Array (Nat × Array (Nat × Dy)))

def toNPoly (p : DPoly) : NPoly :=
  let sorted := (p.toList.mergeSort fun m1 m2 =>
    if m1.a != m2.a then m1.a < m2.a else if m1.b != m2.b then m1.b < m2.b else m1.c ≤ m2.c)
  let rec build : List Mono → NPoly → NPoly
    | [], acc => acc
    | m :: rest, acc =>
      let acc :=
        if h : acc.size > 0 then
          let last := acc[acc.size - 1]
          if last.1 == m.a then
            let inner := last.2
            let inner :=
              if h2 : inner.size > 0 then
                let lb := inner[inner.size - 1]
                if lb.1 == m.b then inner.set! (inner.size - 1) (lb.1, lb.2.push (m.c, m.co))
                else inner.push (m.b, #[(m.c, m.co)])
              else inner.push (m.b, #[(m.c, m.co)])
            acc.set! (acc.size - 1) (last.1, inner)
          else acc.push (m.a, #[(m.b, #[(m.c, m.co)])])
        else acc.push (m.a, #[(m.b, #[(m.c, m.co)])])
      build rest acc
  build sorted #[]

def NPoly.evalIv (p : NPoly) (pu pv pt : Array Iv) : Iv :=
  p.foldl (fun acc (a, ga) =>
    let inner := ga.foldl (fun acc2 (b, gb) =>
      let innermost := gb.foldl (fun acc3 (c, co) => Iv.add acc3 (Iv.scale pt[c]! co)) (Iv.pt Dy.zero)
      Iv.add acc2 (Iv.mul innermost pv[b]!)) (Iv.pt Dy.zero)
    Iv.add acc (Iv.mul inner pu[a]!)) (Iv.pt Dy.zero)

def NPoly.evalPt (p : NPoly) (pu pv pt : Array Dy) : Dy :=
  p.foldl (fun acc (a, ga) =>
    let inner := ga.foldl (fun acc2 (b, gb) =>
      let innermost := gb.foldl (fun acc3 (c, co) => acc3.add (pt[c]!.mul co)) Dy.zero
      acc2.add (innermost.mul pv[b]!)) Dy.zero
    acc.add (inner.mul pu[a]!)) Dy.zero

/-! ## The certificate polynomials -/

namespace Cert

/-- f(u) = sum_k f_k G_k(u), in the first variable. -/
def fPoly (c : Cert) : RPoly :=
  let G := gegenbauer c.d
  (List.range c.d).foldl (fun acc i =>
    let fk := c.f.getD i 0
    if fk == 0 then acc else RPoly.add acc (RPoly.scale (univariate (G.getD (i + 1) []) 0) fk)) []

/-- F(u, v, w) = sum_k <F_k, S_k(u, v, w)>. -/
def FPoly (c : Cert) : RPoly :=
  let LC := legendre c.d
  let TC := chebyshev c.d
  (List.range (c.d + 1)).foldl (fun acc k =>
    let phik := phiPoly k LC
    let Fk := c.F.getD k []
    let n := c.d - k + 1
    (List.range n).foldl (fun acc2 i =>
      (List.range n).foldl (fun acc3 j =>
        let x := (Fk.getD i []).getD j 0
        if x == 0 then acc3 else RPoly.add acc3 (RPoly.scale (sEntry i j phik TC) x)) acc2) acc) []

/-- P1(u) = f(u) + 3 F(1, u, u) + 1, in the first variable. -/
def P1 (c : Cert) (F : RPoly) : RPoly :=
  RPoly.add (RPoly.add c.fPoly (RPoly.scale F.at1uu 3)) ONE

/-- B = 1 + f(1) + F(1, 1, 1). -/
def bound (c : Cert) (F : RPoly) : Rat := 1 + c.fPoly.at111 + F.at111

end Cert

/-! ## The exact half -/

/-- Exact LDL^T over the rationals: true if every pivot is positive. -/
def ldlPD (A : List (List Rat)) : Bool := Id.run do
  let n := A.length
  let mut a : Array (Array Rat) := (A.map List.toArray).toArray
  for k in [0:n] do
    let p := a[k]![k]!
    if p ≤ 0 then return false
    for i in [k+1:n] do
      let l := a[i]![k]! / p
      if l != 0 then
        let mut row := a[i]!
        for j in [k+1:n] do
          row := row.set! j (row[j]! - l * a[k]![j]!)
        a := a.set! i row
  return true

def isSymmetric (A : List (List Rat)) : Bool :=
  let n := A.length
  A.all (fun row => row.length == n) &&
  (List.range n).all fun i => (List.range n).all fun j => (A.getD i []).getD j 0 == (A.getD j []).getD i 0

namespace Cert

/-- The shapes: d numbers f_1..f_d, and F_k of size d - k + 1 for k = 0..d. -/
def shapesOk (c : Cert) : Bool :=
  c.f.length == c.d && c.F.length == c.d + 1 &&
  (List.range (c.d + 1)).all fun k => (c.F.getD k []).length == c.d - k + 1

/-- 1. f_k >= 0 and every F_k symmetric positive definite. -/
def positivityOk (c : Cert) : Bool :=
  c.shapesOk && c.f.all (fun x => 0 ≤ x) && c.F.all (fun A => isSymmetric A && ldlPD A)

/-- 5. The domains run to top >= t, and the tolerances are at most the decimals. -/
def thresholdsOk (c : Cert) : Bool :=
  c.tDec ≤ c.top && 0 ≤ c.e1 && c.e1 ≤ c.e1Dec && 0 ≤ c.e2 && c.e2 ≤ c.e2Dec

/-- 2. (|C| - 1)(1 - e1) - (|C| - 1)(|C| - 2) e2 > B - 1 at |C| = 25, with the decimals. -/
def countOk (c : Cert) (F : RPoly) : Bool :=
  24 * (1 - c.e1Dec) - 552 * c.e2Dec > c.bound F - 1

end Cert

/-- P2 is symmetric in its three variables. -/
def RPoly.symmetric (p : RPoly) : Bool := perms.all fun pm => p.permute pm == p

/-! ## Scaling to dyadic coefficients -/

def oddPart : Nat → Nat → Nat
  | 0, n => n
  | fuel + 1, n => if n > 0 && n % 2 == 0 then oddPart fuel (n / 2) else n

/-- The least odd integer S that makes every coefficient of S p dyadic. -/
def oddScale (p : RPoly) : Nat := p.foldl (fun s (_, q) => Nat.lcm s (oddPart 400 q.den)) 1

/-! ## The branch and bound -/

/-- Boxes narrower than 2^-24 in every direction are not subdivided further. -/
def WMIN : Dy := ⟨1, 24⟩
/-- Hessian bounds are recomputed after this many bisections and inherited in between. -/
def HAGE : Nat := 2

structure Polys where
  P : NPoly
  D : Array NPoly       -- first derivatives
  H : Array NPoly       -- second derivatives, order (0,0),(0,1),(0,2),(1,1),(1,2),(2,2)
  deg : Nat
  E : Dy                -- the scaled tolerance S e
  top : Dy

/-- S p, its derivatives, S e and top, all dyadic; none if some coefficient is not. -/
def mkPolys (p : RPoly) (e top : Rat) : Option Polys :=
  let S : Rat := oddScale p
  match toDPoly (RPoly.scale p S), Dy.ofRat (e * S), Dy.ofRat top with
  | some Pd, some E, some tp =>
    let D := #[Pd.deriv 0, Pd.deriv 1, Pd.deriv 2]
    let H := #[(Pd.deriv 0).deriv 0, (Pd.deriv 0).deriv 1, (Pd.deriv 0).deriv 2,
               (Pd.deriv 1).deriv 1, (Pd.deriv 1).deriv 2, (Pd.deriv 2).deriv 2]
    some ⟨toNPoly Pd, D.map toNPoly, H.map toNPoly, Pd.maxDeg, E, tp⟩
  | _, _, _ => none

inductive Outcome (α : Type) where
  | verified
  | failed
  | split (b1 b2 : α)

/-! ### Constraint (i): one variable -/

structure Box1 where
  lo : Dy
  hi : Dy

def ONEPT : Array Dy := #[Dy.one]
def ONEIV : Array Iv := #[Iv.pt Dy.one]

/-- S e1 - S P1 >= 0 on the interval, by the Taylor form about its centre. -/
def process1 (ps : Polys) (b : Box1) : Outcome Box1 :=
  let c := (b.lo.add b.hi).half
  let r := (b.hi.sub b.lo).half
  let cp := dyPowers c ps.deg
  let pc := ps.P.evalPt cp ONEPT ONEPT
  let g := ((ps.D[0]!).evalPt cp ONEPT ONEPT).abs
  let h := ((ps.H[0]!).evalIv (ivPowers ⟨b.lo, b.hi⟩ ps.deg) ONEIV ONEIV).absHi
  let qlo := ((ps.E.sub pc).sub (g.mul r)).sub (h.mul (r.mul r)).half
  if qlo.isNonneg then .verified
  else if Dy.lt (b.hi.sub b.lo) WMIN then .failed
  else .split ⟨b.lo, c⟩ ⟨c, b.hi⟩

/-- (status, intervals processed): status 0 done, 1 an interval failed, 2 fuel exhausted. -/
def run1 (ps : Polys) (done : Nat) : Nat → List Box1 → Nat × Nat
  | 0, _ => (2, done)
  | _, [] => (0, done)
  | fuel + 1, b :: rest =>
    match process1 ps b with
    | .verified => run1 ps (done + 1) fuel rest
    | .failed => (1, done)
    | .split b1 b2 => run1 ps (done + 1) fuel (b1 :: b2 :: rest)

/-! ### Constraint (ii): three variables, ordered admissible domain -/

structure Box where
  lo : Array Dy
  hi : Array Dy
  hess : Option (Array Dy)
  age : Nat            -- bisections since the Hessian bounds were computed

def hIndex (v w : Nat) : Nat :=
  let (v, w) := if v ≤ w then (v, w) else (w, v)
  if v == 0 then w else if v == 1 then 2 + w else 5

def detPoly : DPoly := #[⟨0, 0, 0, Dy.one⟩, ⟨1, 1, 1, ⟨2, 0⟩⟩, ⟨2, 0, 0, ⟨-1, 0⟩⟩, ⟨0, 2, 0, ⟨-1, 0⟩⟩, ⟨0, 0, 2, ⟨-1, 0⟩⟩]

/-- S e2 - S P2 >= 0 on the part of the box in the ordered admissible domain. -/
def process2 (ps : Polys) (maxAge : Nat) (b : Box) : Outcome Box := Id.run do
  let lo := b.lo; let hi := b.hi
  -- boxes that miss the ordered region u <= v <= w
  if Dy.lt (hi[1]!) (lo[0]!) || Dy.lt (hi[2]!) (lo[1]!) then return .verified
  let ivs : Array Iv := #[⟨lo[0]!, hi[0]!⟩, ⟨lo[1]!, hi[1]!⟩, ⟨lo[2]!, hi[2]!⟩]
  -- boxes on which 1 + 2uvw - u^2 - v^2 - w^2 < 0 throughout
  let pw2 := ivs.map fun x => ivPowers x 2
  let det := detPoly.evalIv (pw2[0]!) (pw2[1]!) (pw2[2]!)
  if Dy.lt det.hi Dy.zero then return .verified
  let c : Array Dy := #[(lo[0]!).add (hi[0]!) |>.half, (lo[1]!).add (hi[1]!) |>.half, (lo[2]!).add (hi[2]!) |>.half]
  let r : Array Dy := #[(hi[0]!).sub (lo[0]!) |>.half, (hi[1]!).sub (lo[1]!) |>.half, (hi[2]!).sub (lo[2]!) |>.half]
  let width := Dy.max (Dy.max ((hi[0]!).sub (lo[0]!)) ((hi[1]!).sub (lo[1]!))) ((hi[2]!).sub (lo[2]!))
  -- Hessian bounds over the box: recomputed every maxAge bisections, inherited from a
  -- containing box in between
  let recompute := b.hess.isNone || b.age ≥ maxAge
  let fresh : Unit → Array Dy := fun _ =>
    let pw := ivs.map fun x => ivPowers x ps.deg
    ps.H.map fun hp => (hp.evalIv (pw[0]!) (pw[1]!) (pw[2]!)).absHi
  let hess : Array Dy :=
    match b.hess with
    | some h => if recompute then fresh () else h
    | none => fresh ()
  let age := if recompute then 0 else b.age
  let cp := c.map fun x => dyPowers x ps.deg
  let q0 := ps.E.sub (ps.P.evalPt (cp[0]!) (cp[1]!) (cp[2]!))
  let mut first : Dy := Dy.zero
  let mut contrib : Array Dy := #[Dy.zero, Dy.zero, Dy.zero]
  for v in [0, 1, 2] do
    let g := ((ps.D[v]!).evalPt (cp[0]!) (cp[1]!) (cp[2]!)).abs
    let term := g.mul (r[v]!)
    first := first.add term
    contrib := contrib.set! v ((contrib[v]!).add term)
  let mut second : Dy := Dy.zero
  for v in [0, 1, 2] do
    for w in [0, 1, 2] do
      if v ≤ w then
        let h := hess[hIndex v w]!
        if v == w then
          let term := h.mul ((r[v]!).mul (r[v]!))
          second := second.add term
          contrib := contrib.set! v ((contrib[v]!).add term)
        else
          let term := (h.mulInt 2).mul ((r[v]!).mul (r[w]!))
          second := second.add term
          contrib := contrib.set! v ((contrib[v]!).add term)
          contrib := contrib.set! w ((contrib[w]!).add term)
  let qlo := (q0.sub first).sub second.half
  if qlo.isNonneg then return .verified
  if Dy.lt width WMIN then return .failed
  -- bisect along the direction of largest contribution
  let axis := if Dy.le (contrib[1]!) (contrib[0]!) && Dy.le (contrib[2]!) (contrib[0]!) then 0
              else if Dy.le (contrib[2]!) (contrib[1]!) then 1 else 2
  let mid := c[axis]!
  return .split ⟨lo, hi.set! axis mid, some hess, age + 1⟩ ⟨lo.set! axis mid, hi, some hess, age + 1⟩

/-- (status, boxes processed): status 0 done, 1 a box failed, 2 fuel exhausted. -/
def run2 (ps : Polys) (hw : Nat) (done : Nat) : Nat → List Box → Nat × Nat
  | 0, _ => (2, done)
  | _, [] => (0, done)
  | fuel + 1, b :: rest =>
    match process2 ps hw b with
    | .verified => run2 ps hw (done + 1) fuel rest
    | .failed => (1, done)
    | .split b1 b2 => run2 ps hw (done + 1) fuel (b1 :: b2 :: rest)

/-! ## The whole check of one certificate -/

structure Checked where
  F : RPoly
  p1 : Option Polys
  p2 : Option Polys

def prepare (c : Cert) : Checked :=
  let F := c.FPoly
  ⟨F, mkPolys (c.P1 F) c.e1 c.top, mkPolys F c.e2 c.top⟩

/-- Constraint (i) on [-1, top]: (status, intervals). -/
def Checked.statI (k : Checked) (fuel : Nat) : Nat × Nat :=
  match k.p1 with
  | none => (3, 0)
  | some ps => run1 ps 0 fuel [⟨⟨-1, 0⟩, ps.top⟩]

/-- Constraint (ii) on the ordered admissible domain in [-1, top]^3: (status, boxes). -/
def Checked.statII (k : Checked) (fuel : Nat) : Nat × Nat :=
  match k.p2 with
  | none => (3, 0)
  | some ps =>
    let m1 : Dy := ⟨-1, 0⟩
    run2 ps HAGE 0 fuel [⟨#[m1, m1, m1], #[ps.top, ps.top, ps.top], none, 0⟩]

/-- 1, 2, 5 and the symmetry of P2: everything but the two branch and bounds. -/
def Checked.exactOk (k : Checked) (c : Cert) : Bool :=
  c.positivityOk && c.thresholdsOk && c.countOk k.F && k.F.symmetric

/-
D4CertDomain.lean: the interval branch and bound of the certificate check,
in exact dyadic arithmetic.

This module re-does step 3 of certificate_check.py inside Lean: from the
certificate (f, F_0, ..., F_8) of D4CertData it expands the polynomial
P(u, v, t) exactly over the rationals, scales it by 315 so that every
coefficient is a dyadic rational, and then verifies, by a depth-first
branch and bound over boxes with the second-order Taylor form, that

    315 * 1000 * (omega(u) + omega(v) + omega(t)) - 315 * P(u, v, t) >= 0

on the ordered admissible domain -1 <= u <= v <= t <= 1/2,
1 + 2uvt - u^2 - v^2 - t^2 >= 0.  The only inputs that are not computed
here are the tables of D4CertData for omega, omega' and omega'' (dyadic
lower and upper bounds evaluated from the closed forms in interval
arithmetic by certificate_check.py); everything else, the expansion of P,
its derivatives, the enclosures and the subdivision, is exact integer
arithmetic on numerators and exponents, with no rounding anywhere.  The
theorem at the end is settled by native_decide, so it trusts the Lean
compiler as well as the kernel.
-/
import D4CertData

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

def Fmats : List (List (List Rat)) := [F0, F1, F2, F3, F4, F5, F6, F7, F8]

/-- P = f(u)+f(v)+f(t) + F(u,v,t) + (F(1,u,u)+F(1,v,v)+F(1,t,t))/21. -/
def buildP (d : Nat) : RPoly :=
  let LC := legendre d
  let G := gegenbauer d
  let TC := chebyshev d
  let fpart := (List.range (d + 1)).foldl (fun acc k =>
    let fk := f.getD k 0
    if fk == 0 then acc else
    (List.range 3).foldl (fun acc2 var => RPoly.add acc2 (RPoly.scale (univariate (G.getD k []) var) fk)) acc) []
  (List.range (d + 1)).foldl (fun acc k =>
    let phik := phiPoly k LC
    let Fk := Fmats.getD k []
    let n := d - k + 1
    (List.range n).foldl (fun acc2 i =>
      (List.range n).foldl (fun acc3 j =>
        let c := (Fk.getD i []).getD j 0
        if c == 0 then acc3 else
        let s := sEntry i j phik TC
        let s1 := s.at1uu
        let coinc := RPoly.add (RPoly.add s1 (s1.permute (1, 0, 2))) (s1.permute (2, 1, 0))
        RPoly.add acc3 (RPoly.add (RPoly.scale s c) (RPoly.scale coinc (c / 21)))) acc2) acc) fpart

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

/-! ## The tables -/

def mkDy (x : Int × Nat) : Dy := ⟨x.1, x.2⟩
def usT : Array Dy := usRaw.map mkDy
def olowT : Array Dy := olowRaw.map mkDy
def d1loT : Array Dy := d1loRaw.map mkDy
def d1hiT : Array Dy := d1hiRaw.map mkDy
def duT : Dy := mkDy duRaw
def m2T : Dy := mkDy m2Raw

/-- A table of omega: at the points us[i] (increasing), olow[i] <= omega and
d1lo[i] <= omega' <= d1hi[i], with omega increasing; du bounds the distance from a
point to the table point below it, and m2 bounds |omega''| where the table is used. -/
structure OTab where
  us : Array Dy
  olow : Array Dy
  d1lo : Array Dy
  d1hi : Array Dy
  du : Dy
  m2 : Dy

/-- The tables of D4CertData, from certificate_check.py. -/
def dataTab : OTab := ⟨usT, olowT, d1loT, d1hiT, duT, m2T⟩

/-- The largest index i with us[i] <= x, or none if x < us[0]. -/
def lookupIn (us : Array Dy) (x : Dy) : Option Nat :=
  if Dy.lt x (us[0]!) then none else
  let rec go (lo hi : Nat) : Nat → Nat
    | 0 => lo
    | fuel + 1 =>
      if hi - lo ≤ 1 then lo else
      let mid := (lo + hi) / 2
      if Dy.le (us[mid]!) x then go mid hi fuel else go lo mid fuel
  some (go 0 us.size 64)

def lookup (x : Dy) : Option Nat := lookupIn usT x

/-! ## The branch and bound -/

/-- 315 * 1000: the scale of the omega terms after the scaling of P by 315. -/
def SC : Int := 315000
/-- A dyadic number above 1/3. -/
def THIRD : Dy := ⟨5592406, 24⟩
/-- Boxes narrower than 2^-20 in every direction are not subdivided further. -/
def WMIN : Dy := ⟨1, 20⟩
/-- Hessian bounds are recomputed after this many bisections and inherited in between. -/
def HAGE : Nat := 2

structure Polys where
  P : NPoly
  D : Array NPoly       -- three first derivatives
  H : Array NPoly       -- six second derivatives, order (0,0),(0,1),(0,2),(1,1),(1,2),(2,2)
  deg : Nat

structure Box where
  lo : Array Dy
  hi : Array Dy
  hess : Option (Array Dy)
  age : Nat            -- bisections since the Hessian bounds were computed

inductive Outcome where
  | verified
  | failed
  | split (b1 b2 : Box)

def hIndex (v w : Nat) : Nat :=
  let (v, w) := if v ≤ w then (v, w) else (w, v)
  if v == 0 then w else if v == 1 then 2 + w else 5

def detPoly : DPoly := #[⟨0, 0, 0, Dy.one⟩, ⟨1, 1, 1, ⟨2, 0⟩⟩, ⟨2, 0, 0, ⟨-1, 0⟩⟩, ⟨0, 2, 0, ⟨-1, 0⟩⟩, ⟨0, 0, 2, ⟨-1, 0⟩⟩]

def processWith (tab : OTab) (ps : Polys) (maxAge : Nat) (b : Box) : Outcome := Id.run do
  let lo := b.lo; let hi := b.hi
  -- the ordered region u <= v <= t
  if Dy.lt (hi[1]!) (lo[0]!) || Dy.lt (hi[2]!) (lo[1]!) then return .verified
  let ivs : Array Iv := #[⟨lo[0]!, hi[0]!⟩, ⟨lo[1]!, hi[1]!⟩, ⟨lo[2]!, hi[2]!⟩]
  let pw2 := ivs.map fun x => ivPowers x 2
  let det := detPoly.evalIv (pw2[0]!) (pw2[1]!) (pw2[2]!)
  if Dy.lt det.hi Dy.zero then return .verified
  -- centre and half-widths
  let c : Array Dy := #[(lo[0]!).add (hi[0]!) |>.half, (lo[1]!).add (hi[1]!) |>.half, (lo[2]!).add (hi[2]!) |>.half]
  let r : Array Dy := #[(hi[0]!).sub (lo[0]!) |>.half, (hi[1]!).sub (lo[1]!) |>.half, (hi[2]!).sub (lo[2]!) |>.half]
  let width := Dy.max (Dy.max ((hi[0]!).sub (lo[0]!)) ((hi[1]!).sub (lo[1]!))) ((hi[2]!).sub (lo[2]!))
  -- Hessian bounds: recomputed every maxAge bisections, inherited in between
  let recompute := b.hess.isNone || b.age ≥ maxAge
  let hess : Array Dy :=
    match b.hess with
    | some h => if !recompute then h else
        let pw := ivs.map fun x => ivPowers x ps.deg
        ps.H.map fun hp => (hp.evalIv (pw[0]!) (pw[1]!) (pw[2]!)).absHi
    | none =>
        let pw := ivs.map fun x => ivPowers x ps.deg
        ps.H.map fun hp => (hp.evalIv (pw[0]!) (pw[1]!) (pw[2]!)).absHi
  let age := if recompute then 0 else b.age
  -- omega terms at the centre
  let mut q0 : Dy := Dy.zero
  let mut glo : Array Dy := #[Dy.zero, Dy.zero, Dy.zero]
  let mut ghi : Array Dy := #[Dy.zero, Dy.zero, Dy.zero]
  let mut m2v : Array Dy := #[Dy.zero, Dy.zero, Dy.zero]
  let dM := tab.m2.mul tab.du
  for v in [0, 1, 2] do
    if Dy.le THIRD (lo[v]!) then
      match lookupIn tab.us (c[v]!) with
      | none => pure ()
      | some idx =>
        q0 := q0.add ((tab.olow[idx]!).mulInt SC)
        glo := glo.set! v (((tab.d1lo[idx]!).sub dM).mulInt SC)
        ghi := ghi.set! v (((tab.d1hi[idx]!).add dM).mulInt SC)
        m2v := m2v.set! v (tab.m2.mulInt SC)
  -- P and its gradient at the centre
  let cp := c.map fun x => dyPowers x ps.deg
  let pc := ps.P.evalPt (cp[0]!) (cp[1]!) (cp[2]!)
  q0 := q0.sub pc
  let mut first : Dy := Dy.zero
  let mut contrib : Array Dy := #[Dy.zero, Dy.zero, Dy.zero]
  let mut gabs : Array Dy := #[Dy.zero, Dy.zero, Dy.zero]
  for v in [0, 1, 2] do
    let g := (ps.D[v]!).evalPt (cp[0]!) (cp[1]!) (cp[2]!)
    let dqlo := (glo[v]!).sub g
    let dqhi := (ghi[v]!).sub g
    let ga := Dy.max dqlo.abs dqhi.abs
    gabs := gabs.set! v ga
    let term := ga.mul (r[v]!)
    first := first.add term
    contrib := contrib.set! v ((contrib[v]!).add term)
  let mut second : Dy := Dy.zero
  for v in [0, 1, 2] do
    for w in [0, 1, 2] do
      if v ≤ w then
        let h := hess[hIndex v w]!
        if v == w then
          let term := (h.add (m2v[v]!)).mul ((r[v]!).mul (r[v]!))
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
  let b1 : Box := ⟨lo, hi.set! axis mid, some hess, age + 1⟩
  let b2 : Box := ⟨lo.set! axis mid, hi, some hess, age + 1⟩
  return .split b1 b2

def process (ps : Polys) (maxAge : Nat) (b : Box) : Outcome := processWith dataTab ps maxAge b

/-- The branch and bound with a given table of omega: (status, boxes). -/
def runStatWith (tab : OTab) (ps : Polys) (hw : Nat) (done : Nat) : Nat → List Box → Nat × Nat
  | 0, _ => (2, done)
  | _, [] => (0, done)
  | fuel + 1, b :: rest =>
    match processWith tab ps hw b with
    | .verified => runStatWith tab ps hw (done + 1) fuel rest
    | .failed => (1, done)
    | .split b1 b2 => runStatWith tab ps hw (done + 1) fuel (b1 :: b2 :: rest)

def run (ps : Polys) (hw : Nat) : Nat → List Box → Bool
  | 0, _ => false
  | _, [] => true
  | fuel + 1, b :: rest =>
    match process ps hw b with
    | .verified => run ps hw fuel rest
    | .failed => false
    | .split b1 b2 => run ps hw fuel (b1 :: b2 :: rest)

/-- Diagnostic version: (status, boxes processed), status 0 done, 1 a box
failed, 2 fuel exhausted. -/
def runStat (ps : Polys) (hw : Nat) (done : Nat) : Nat → List Box → Nat × Nat
  | 0, _ => (2, done)
  | _, [] => (0, done)
  | fuel + 1, b :: rest =>
    match process ps hw b with
    | .verified => runStat ps hw (done + 1) fuel rest
    | .failed => (1, done)
    | .split b1 b2 => runStat ps hw (done + 1) fuel (b1 :: b2 :: rest)

/-- The whole check: expand P, scale by 315, differentiate, and run the
branch and bound from the box [-1, 1/2]^3. -/
def thePolys : Option Polys :=
  let P := buildP 8
  match toDPoly (RPoly.scale P 315) with
  | none => none
  | some Pd =>
    let D := #[Pd.deriv 0, Pd.deriv 1, Pd.deriv 2]
    let H := #[(Pd.deriv 0).deriv 0, (Pd.deriv 0).deriv 1, (Pd.deriv 0).deriv 2,
               (Pd.deriv 1).deriv 1, (Pd.deriv 1).deriv 2, (Pd.deriv 2).deriv 2]
    some ⟨toNPoly Pd, D.map toNPoly, H.map toNPoly, Pd.maxDeg⟩

def startBox : Box := ⟨#[⟨-1, 0⟩, ⟨-1, 0⟩, ⟨-1, 0⟩], #[⟨1, 1⟩, ⟨1, 1⟩, ⟨1, 1⟩], none, 0⟩

def verifyWithFuel (fuel : Nat) : Bool :=
  match thePolys with
  | none => false
  | some ps => run ps HAGE fuel [startBox]

/-- The whole check with a given table of omega: (status, boxes). -/
def verifyStatWith (tab : OTab) (fuel : Nat) : Nat × Nat :=
  match thePolys with
  | none => (3, 0)
  | some ps => runStatWith tab ps HAGE 0 fuel [startBox]

def verifyStat (fuel : Nat) : Nat × Nat :=
  match thePolys with
  | none => (3, 0)
  | some ps => runStat ps HAGE 0 fuel [startBox]

/-- Diagnostic: the check on a given box, with a given width threshold for
recomputing the Hessian bounds; box coordinates as numerators over 2^20. -/
def verifyStatBox (fuel : Nat) (maxAge : Nat) (lo hi : List Int) : Nat × Nat :=
  match thePolys with
  | none => (3, 0)
  | some ps =>
    let mk (l : List Int) : Array Dy := (l.map fun n => (⟨n, 20⟩ : Dy)).toArray
    runStat ps maxAge 0 fuel [⟨mk lo, mk hi, none, 0⟩]


/-- Number of monomials of the expanded P, for the record. -/
def monomialCount : Nat := (buildP 8).length

/-
Kernel checker for the certificate of the two-translate packing problem in R^4.
Core Lean only. Matrices are lists of rows of integers; `ent A i j` is the entry (i, j),
`0` outside the list. Every function returns a `Bool`; `D4Lean.Sound` proves what a
`true` result means.
-/
namespace D4

abbrev Mat := List (List Int)

/-- Entry `(i, j)` of a list matrix. -/
def ent (A : Mat) (i j : Nat) : Int := (A.getD i []).getD j 0

/-- Coordinate `i` of a list vector. -/
def vg (v : List Int) (i : Nat) : Int := v.getD i 0

/-- `sumN n f = f 0 + ⋯ + f (n - 1)`. -/
def sumN : Nat → (Nat → Int) → Int
  | 0, _ => 0
  | n + 1, f => sumN n f + f n

/-- `prodN n f = f 0 * ⋯ * f (n - 1)`. -/
def prodN : Nat → (Nat → Int) → Int
  | 0, _ => 1
  | n + 1, f => prodN n f * f n

/-- `allN n p` holds when `p i` holds for every `i < n`. -/
def allN : Nat → (Nat → Bool) → Bool
  | 0, _ => true
  | n + 1, p => allN n p && p n

/-- Dot product of two lists (truncated to the shorter one). -/
def dot : List Int → List Int → Int
  | a :: as, b :: bs => a * b + dot as bs
  | _, _ => 0

/-- The quadratic form `J[k]`, computed by dot products. -/
def qf (A : Mat) (k : List Int) : Int := dot k (A.map fun r => dot r k)

/-- `A` has `n` rows, each of length `n`. -/
def shape (A : Mat) (n : Nat) : Bool := A.length == n && A.all fun r => r.length == n

def symm (A : Mat) : Bool := allN 5 fun i => allN 5 fun j => ent A i j == ent A j i

/-- Entry `(i, j)` of the product `A * B` of `n × n` matrices. -/
def mulE (n : Nat) (A B : Mat) (i j : Nat) : Int := sumN n fun k => ent A i k * ent B k j

/-- Entry `(i, j)` of `Tᵀ * A * T`. -/
def congE (T A : Mat) (i j : Nat) : Int :=
  sumN 5 fun a => sumN 5 fun b => ent T a i * ent A a b * ent T b j

/-! ### Membership in `R` by a box enumeration -/

def rangeAll (p : Int → Bool) : Int → Nat → Bool
  | _, 0 => true
  | v, k + 1 => p v && rangeAll p (v + 1) k

/-- `boxAll [(lo₁, n₁), …] p`: `p` holds at every integer point with `lo_i ≤ x_i < lo_i + n_i`. -/
def boxAll : List (Int × Nat) → (List Int → Bool) → Bool
  | [], p => p []
  | (lo, n) :: rest, p => rangeAll (fun v => boxAll rest (fun xs => p (v :: xs))) lo n

def negV (k : List Int) : List Int := k.map fun x => -x

def isZero (k : List Int) : Bool := k.all fun x => x == 0

/-- At a lattice point of the box: `k = 0`, or `J[k] > 4`, or `J[k] = 4` and `±k` is listed. -/
def ptOK (J : Mat) (mins : List (List Int)) (k : List Int) : Bool :=
  isZero k || decide (4 < qf J k) || (qf J k == 4 && (mins.contains k || mins.contains (negV k)))

/-- A listed minimal vector: of length 5, last coordinate 0 or 1, nonzero, and `J[k] = 4`. -/
def minOK (J : Mat) (k : List Int) : Bool :=
  k.length == 5 && (vg k 4 == 0 || vg k 4 == 1) && !isZero k && qf J k == 4

/-- The bound certificate for coordinate `i`: `C · J' = ∑ e_r m_r m_rᵀ` with `e_r ≥ 0`,
where `J' = cd · J + cn · E₄₄`, the first term is `(e, unit vector i)` and
`e (b + 1)² > C (4 cd + cn)`. -/
def boundOK (J : Mat) (cd cn C : Int) (i : Nat) (b : Nat)
    (terms : List (Int × List Int)) : Bool :=
  match terms with
  | [] => false
  | (e, m) :: _ =>
    m.length == 5 && (allN 5 fun a => vg m a == if a == i then 1 else 0) &&
    decide (C * (4 * cd + cn) < e * ((b : Int) + 1) * ((b : Int) + 1)) &&
    (terms.all fun t => decide (0 ≤ t.1) && t.2.length == 5) &&
    (allN 5 fun a => allN 5 fun c =>
      C * (cd * ent J a c + (if a == 4 && c == 4 then cn else 0)) ==
        (terms.map fun t => t.1 * vg t.2 a * vg t.2 c).foldr (· + ·) 0)

/-- The box `[-b_i, b_i]` in the first four coordinates and `{0, 1}` in the last. -/
def boxOf (bs : List Nat) : List (Int × Nat) :=
  bs.map (fun (b : Nat) => ((-(b : Int)), 2 * b + 1)) ++ [((0 : Int), (2 : Nat))]

/-- `J ∈ R`, with `mins` the minimal vectors: the four bound certificates and the box. -/
def memOK (J : Mat) (mins : List (List Int)) (cd cn C : Int) (bs : List Nat)
    (certs : List (List (Int × List Int))) : Bool :=
  decide (0 < cd) && decide (0 ≤ cn) && decide (0 < C) && bs.length == 4 && certs.length == 4 &&
  (allN 4 fun i => boundOK J cd cn C i (bs.getD i 0) (certs.getD i [])) &&
  mins.all (minOK J) && boxAll (boxOf bs) (ptOK J mins)

/-! ### Determinants by triangular elimination -/

/-- `E` is lower triangular, `E * A` is upper triangular, `∏ diag(E A) = d ∏ diag E` and the same
for the leading `4 × 4` blocks with `dq`, and `∏ diag E ≠ 0`. Then `det A = d`, `det A₄ = dq`. -/
def detOK (A E : Mat) (d dq : Int) : Bool :=
  (allN 5 fun i => allN 5 fun j => decide (j ≤ i) || ent E i j == 0) &&
  (allN 5 fun i => allN 5 fun j => decide (i ≤ j) || mulE 5 E A i j == 0) &&
  prodN 5 (fun i => ent E i i) != 0 &&
  prodN 5 (fun i => mulE 5 E A i i) == d * prodN 5 (fun i => ent E i i) &&
  prodN 4 (fun i => mulE 5 E A i i) == dq * prodN 4 (fun i => ent E i i)

/-! ### Polynomials: coefficient lists, lowest degree first -/

def powI (w : Int) : Nat → Int
  | 0 => 1
  | n + 1 => powI w n * w

/-- `hom p y w n = ∑ p_i y^i w^(n-i)`, the value `w^n p(y / w)`. -/
def hom : List Int → Int → Int → Nat → Int
  | [], _, _, _ => 0
  | a :: r, y, w, n => a * powI w n + y * hom r y w (n - 1)

def addP : List Int → List Int → List Int
  | [], q => q
  | p, [] => p
  | a :: p, b :: q => (a + b) :: addP p q

/-- Coefficients of `(u + h x) c(x)`. -/
def mulLin (u h : Int) (c : List Int) : List Int := addP (c.map (u * ·)) (0 :: c.map (h * ·))

/-- Coefficients in `x` of `hom p (u + h x) w n`. -/
def comp : List Int → Int → Int → Int → Nat → List Int
  | [], _, _, _, _ => []
  | a :: r, u, h, w, n => addP [a * powI w n] (mulLin u h (comp r u h w (n - 1)))

def negPart : List Int → Int
  | [] => 0
  | a :: r => min a 0 + negPart r

/-- Positive on `(0, 1]`: after the leading zeros, `c_m + ∑_{i > m} min(c_i, 0) > 0`. -/
def posBound : List Int → Bool
  | [] => false
  | a :: r => if a == 0 then posBound r else decide (0 < a + negPart r)

/-- Positive on `(0, ∞)`: all coefficients `≥ 0` and the first nonzero one `> 0`. -/
def posInf : List Int → Bool
  | [] => false
  | a :: r => if a == 0 then posInf r else decide (0 < a) && r.all fun c => decide (0 ≤ c)

/-- `g = det Q - 256` from the coefficients of `det Q`. -/
def gOf : List Int → List Int
  | [] => [-256]
  | a :: r => (a - 256) :: r

/-! ### Edges -/

/-- The value of the polynomial `p` (degree bound `n`) or of `g` on a piece. -/
def pieceOK (pJ g : List Int) (wd : Int) (modeJ dirL : Bool) (u v : Int) : Bool :=
  let P := if modeJ then pJ else g
  let n := if modeJ then 5 else 4
  if dirL then posBound (comp P u (v - u) wd n) else posBound (comp P v (u - v) wd n)

/-- At the breakpoint `t = u / wd`: `det J ≤ 0 → det Q ≥ 256`, and `det J = 0 ∧ det Q = 256`
only at a listed point. -/
def pointOK (pJ g : List Int) (wd : Int) (eqs : List Int) (u : Int) : Bool :=
  let a := hom pJ u wd 5
  let b := hom g u wd 4
  (decide (0 < a) || decide (0 ≤ b)) && (!(a == 0 && b == 0) || eqs.contains u)

def sorted : List Int → Bool
  | a :: b :: r => decide (a < b) && sorted (b :: r)
  | _ => true

/-- Consecutive breakpoints and the pieces between them. -/
def piecesOK (pJ g : List Int) (wd : Int) : List Int → List (Bool × Bool) → Bool
  | u :: v :: r, (m, d) :: ps => pieceOK pJ g wd m d u v && piecesOK pJ g wd (v :: r) ps
  | [_], [] => true
  | _, _ => false

/-- The sample points `t_j = sn_j / sd` with their elimination certificates for `sd J + sn_j Y`. -/
def samplesOK (J Y : Mat) (pJ pQ : List Int) (sd : Int) (sn : List Int) (Es : List Mat) : Bool :=
  decide (0 < sd) && sn.length == 6 && Es.length == 6 && sorted sn &&
  (allN 6 fun j =>
    let s := vg sn j
    detOK ((List.range 5).map fun a => (List.range 5).map fun b => sd * ent J a b + s * ent Y a b)
      (Es.getD j []) (hom pJ s sd 5) (hom pQ s sd 4))

/-- The sign certificate along an edge: breakpoints `us / wd` from `0`, pieces between them. -/
def signOK (pJ pQ : List Int) (wd : Int) (us : List Int) (ps : List (Bool × Bool))
    (eqs : List Int) : Bool :=
  let g := gOf pQ
  decide (0 < wd) && pJ.length ≤ 6 && pQ.length ≤ 5 && us.head? == some 0 && sorted us &&
  us.all (pointOK pJ g wd eqs) && piecesOK pJ g wd us ps

/-- A bounded edge `J + tY`, `0 ≤ t ≤ tn / td`: the end point is `Tᵀ J_to T` with `T ∈ Γ`
(inverse `Tinv`), and `kmax` is a minimal vector of the end point with `Y[kmax] < 0`. -/
def boundedOK (J Y Jto T Tinv : Mat) (tn td : Int) (kmax : List Int) : Bool :=
  shape Y 5 && symm Y && shape T 5 && shape Tinv 5 && decide (0 < tn) && decide (0 < td) &&
  (allN 5 fun i => allN 5 fun j => td * ent J i j + tn * ent Y i j == td * congE T Jto i j) &&
  (allN 5 fun i => allN 5 fun j => mulE 5 T Tinv i j == if i == j then 1 else 0) &&
  (allN 4 fun j => ent T 4 j == 0) && (ent T 4 4 == 1 || ent T 4 4 == -1) &&
  kmax.length == 5 && (vg kmax 4 == 0 || vg kmax 4 == 1) && !isZero kmax &&
  td * qf J kmax + tn * qf Y kmax == 4 * td && decide (qf Y kmax < 0)

/-- An unbounded edge: `Y = [[λ w wᵀ, μ w], [μ wᵀ, s]]` with `λ n² + 2 μ n + s ≥ 0` on `ℤ`
(checked at `n₀ = ⌊-μ/λ⌋` and `n₀ + 1`), so `Y[k] ≥ 0` for every `k`. -/
def unboundedOK (Y : Mat) (lam mu s n0 : Int) (w : List Int) : Bool :=
  shape Y 5 && symm Y && w.length == 4 &&
  (allN 4 fun a => allN 4 fun b => ent Y a b == lam * vg w a * vg w b) &&
  (allN 4 fun a => ent Y a 4 == mu * vg w a) && ent Y 4 4 == s && decide (0 ≤ lam) &&
  (if lam == 0 then mu == 0 && decide (0 ≤ s) else
    decide (lam * n0 ≤ -mu) && decide (-mu < lam * (n0 + 1)) &&
    decide (0 ≤ lam * n0 * n0 + 2 * mu * n0 + s) &&
    decide (0 ≤ lam * (n0 + 1) * (n0 + 1) + 2 * mu * (n0 + 1) + s))

/-- The sign certificate for an unbounded edge ends with the piece `[u_last, ∞)`. -/
def tailOK (pJ pQ : List Int) (wd : Int) (us : List Int) (modeJ : Bool) : Bool :=
  match us.getLast? with
  | none => false
  | some u => if modeJ then posInf (comp pJ u 1 wd 5) else posInf (comp (gOf pQ) u 1 wd 4)

/-! ### The certificate records -/

structure Vtx where
  J : Mat
  mins : List (List Int)
  cd : Int
  cn : Int
  C : Int
  bs : List Nat
  certs : List (List (Int × List Int))
  E : Mat
  dJ : Int
  dQ : Int
  eq : Bool
deriving Inhabited

structure Edge where
  Y : Mat
  bounded : Bool
  tn : Int
  td : Int
  target : Nat
  T : Mat
  Tinv : Mat
  kmax : List Int
  lam : Int
  mu : Int
  s : Int
  n0 : Int
  w : List Int
  pJ : List Int
  pQ : List Int
  sd : Int
  sn : List Int
  Es : List Mat
  wd : Int
  us : List Int
  ps : List (Bool × Bool)
  tailJ : Bool
  eqs : List Int
deriving Inhabited

def Vtx.ok (v : Vtx) : Bool :=
  shape v.J 5 && symm v.J && memOK v.J v.mins v.cd v.cn v.C v.bs v.certs &&
  detOK v.J v.E v.dJ v.dQ &&
  (decide (0 < v.dJ) || decide (256 ≤ v.dQ)) && ((v.dJ == 0 && v.dQ == 256) == v.eq)

def Edge.ok (J Jto : Mat) (e : Edge) : Bool :=
  samplesOK J e.Y e.pJ e.pQ e.sd e.sn e.Es && signOK e.pJ e.pQ e.wd e.us e.ps e.eqs &&
  if e.bounded then
    boundedOK J e.Y Jto e.T e.Tinv e.tn e.td e.kmax && e.us.getLast? == some (e.tn * e.wd / e.td) &&
      e.tn * e.wd % e.td == 0
  else
    unboundedOK e.Y e.lam e.mu e.s e.n0 e.w && tailOK e.pJ e.pQ e.wd e.us e.tailJ

/-! ### The cone at a vertex

A symmetric `5 × 5` matrix `X` is written by its upper triangle `psiL X`, in the order
`(0,0), (0,1), …, (0,4), (1,1), …, (4,4)`; then `X[k] = symcL k ⬝ psiL X`. The rows of a vertex
are `symcL k` for its minimal vectors `k`. -/

/-- The positions of the upper triangle. -/
def pairs : List (Nat × Nat) :=
  [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 1), (1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4),
    (3, 3), (3, 4), (4, 4)]

/-- The upper triangle of a list matrix. -/
def psiL (Y : Mat) : List Int := pairs.map fun p => ent Y p.1 p.2

/-- The coefficients of `X ↦ X[k]` on the upper triangle. -/
def symcL (k : List Int) : List Int :=
  pairs.map fun p => (if p.1 == p.2 then 1 else 2) * vg k p.1 * vg k p.2

/-- The rank certificate: `∑ₜ B u t · rows[idx t] = c eᵤ` with `c ≠ 0`, so the rows span. -/
def rankOK (rows : List (List Int)) (idx : List Nat) (B : Mat) (c : Int) : Bool :=
  c != 0 && idx.length == 15 && idx.all (· < rows.length) &&
  (allN 15 fun u => allN 15 fun v =>
    sumN 15 (fun t => ent B u t * vg (rows.getD (idx.getD t 0) []) v) == if u == v then c else 0)

/-- How a hyperplane flat is certified: its kernel vector `y` takes both signs on the rows `p`
and `q`, or `y` is the listed ray `r` (or its negative). -/
inductive FKind where
  | mixed (y : List Int) (p q : Nat)
  | ray (r : Nat) (neg : Bool)
deriving Inhabited

/-- A set of rows, given by the bits of `mask`, and the certificate of its kernel vector. -/
structure Flat where
  mask : Nat
  kind : FKind
deriving Inhabited

/-- Flats in a Braun tree, so that the `n`-th one is found in `O(log n)` steps. -/
inductive FTree where
  | nil
  | node (l : FTree) (f : Flat) (r : FTree)
deriving Inhabited

def FTree.get : FTree → Nat → Option Flat
  | .nil, _ => none
  | .node _ f _, 0 => some f
  | .node l _ r, n + 1 => if n % 2 == 0 then l.get (n / 2) else r.get (n / 2)

/-- The kernel vector of a flat. -/
def flatY (rays : List (List Int)) (f : Flat) : List Int :=
  match f.kind with
  | .mixed y _ _ => y
  | .ray r neg => if neg then negV (rays.getD r []) else rays.getD r []

/-- The kernel vector is nonzero, vanishes on the rows of the flat, and takes both signs on the
rows or is a listed ray. -/
def flatOK (rows rays : List (List Int)) (f : Flat) : Bool :=
  let y := flatY rays f
  y.length == 15 && !isZero y &&
  (allN rows.length fun i => !(f.mask.testBit i) || dot (rows.getD i []) y == 0) &&
  match f.kind with
  | .mixed _ p q => decide (p < rows.length) && decide (q < rows.length) &&
      decide (0 < dot (rows.getD p []) y) && decide (dot (rows.getD q []) y < 0)
  | .ray r _ => decide (r < rays.length)

/-- The `k`-element sublists of a list. -/
def combs : Nat → List Nat → List (List Nat)
  | 0, _ => [[]]
  | _ + 1, [] => []
  | k + 1, x :: xs => (combs k xs).map (x :: ·) ++ combs (k + 1) xs

/-- Walk through the `k`-element sublists `S` of `l` in the order of `combs`, the rows in the mask
`acc` having been chosen before: the rows of `acc` and of `S` must lie in the flat `t.get j` for
the next index `j` of `js`. The result is the list of the indices that are left. -/
def subsOK (t : FTree) : Nat → List Nat → Nat → List Nat → Option (List Nat)
  | 0, _, acc, j :: js =>
    match t.get j with
    | some f => if acc &&& f.mask == acc then some js else none
    | none => none
  | 0, _, _, [] => none
  | _ + 1, [], _, js => some js
  | k + 1, x :: xs, acc, js =>
    match subsOK t k xs (acc ||| 2 ^ x) js with
    | some js' => subsOK t (k + 1) xs acc js'
    | none => none

theorem subsOK_append (t : FTree) : ∀ (k : Nat) (l : List Nat) (acc : Nat) (js rest B : List Nat),
    subsOK t k l acc js = some rest → subsOK t k l acc (js ++ B) = some (rest ++ B) := by
  intro k
  induction k with
  | zero =>
    intro l acc js rest B h
    cases js with
    | nil => cases l <;> simp [subsOK] at h
    | cons j js =>
      cases hg : t.get j with
      | none => cases l <;> simp [subsOK, hg] at h
      | some f =>
        cases l <;> simp only [subsOK, hg, List.cons_append] at h ⊢ <;> split at h <;>
          simp_all
  | succ k ihk =>
    intro l
    induction l with
    | nil => intro acc js rest B h; simp only [subsOK, Option.some.injEq] at h ⊢; rw [h]
    | cons x xs ihl =>
      intro acc js rest B h
      simp only [subsOK] at h ⊢
      cases h1 : subsOK t k xs (acc ||| 2 ^ x) js with
      | none => simp [h1] at h
      | some js' =>
        rw [h1] at h
        rw [ihk xs _ js js' B h1]
        exact ihl acc js' rest B h

/-- The sets of rows are checked in pieces: those containing the next row `x`, and the others. -/
theorem subsOK_node {t : FTree} {k x : Nat} {xs : List Nat} {acc : Nat}
    (h1 : ∃ A, subsOK t k xs (acc ||| 2 ^ x) A = some [])
    (h2 : ∃ B, subsOK t (k + 1) xs acc B = some []) :
    ∃ js, subsOK t (k + 1) (x :: xs) acc js = some [] := by
  obtain ⟨A, hA⟩ := h1
  obtain ⟨B, hB⟩ := h2
  refine ⟨A ++ B, ?_⟩
  have h := subsOK_append t k xs _ A [] B hA
  rw [List.nil_append] at h
  simp only [subsOK, h, hB]

/-! The flats and their indices are stored as natural numbers. -/

/-- The `k` lowest digits of `n` in base `b`. -/
def digits (b : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | k + 1, n => n % b :: digits b k (n / b)

/-- The first `n` flat indices, stored 64 to a number in base `4096`. -/
def idxs (ps : List Nat) (n : Nat) : List Nat := (ps.flatMap (digits 4096 64)).take n

/-- A flat stored in a natural number: the mask in the lowest 20 bits, then a tag bit. For a ray
follow the sign and the index; for a mixed flat the rows `p` and `q` (5 bits each) and the 15
coordinates of `y`, each in `w` bits and offset by `2 ^ (w - 1)`. -/
def unpackFlat (w n : Nat) : Flat :=
  let mask := n % 2 ^ 20
  let r := n / 2 ^ 20
  if r % 2 == 0 then ⟨mask, .ray (r / 4) (r / 2 % 2 == 1)⟩
  else
    let s := r / 2
    ⟨mask, .mixed ((digits (2 ^ w) 15 (s / 1024)).map fun d : Nat => (d : Int) - (2 : Int) ^ (w - 1))
      (s % 32) (s / 32 % 32)⟩

/-- The elements at even and at odd positions. -/
def split {α : Type} : List α → List α × List α
  | [] => ([], [])
  | [x] => ([x], [])
  | x :: y :: r => ((x :: (split r).1), (y :: (split r).2))

/-- The Braun tree of a list (of depth at most `k`): `(ofList k l).get n` is the `n`-th element. -/
def FTree.ofList : Nat → List Flat → FTree
  | 0, _ => .nil
  | _ + 1, [] => .nil
  | k + 1, f :: fs => .node (FTree.ofList k (split fs).1) f (FTree.ofList k (split fs).2)

/-- The cone data of a vertex: the rank certificate and the flats. -/
structure ConeData where
  idx : List Nat
  B : Mat
  c : Int
  w : Nat
  flats : List Nat
deriving Inhabited

/-- The flats of the cone data. -/
def ConeData.tree (cd : ConeData) : FTree := FTree.ofList 20 (cd.flats.map (unpackFlat cd.w))

/-- Every listed ray `y` has `y ⬝ a ≥ 0` for every row `a`. -/
def raysOK (rows rays : List (List Int)) : Bool :=
  rays.all fun y => y.length == 15 && rows.all fun a => decide (0 ≤ dot a y)

/-- Every flat stored in `fl` is certified. -/
def flatsOK (rows rays : List (List Int)) (w : Nat) (fl : List Nat) : Bool :=
  fl.all fun n => flatOK rows rays (unpackFlat w n)

theorem raysOK_append (rows a b : List (List Int)) :
    raysOK rows (a ++ b) = (raysOK rows a && raysOK rows b) := by
  simp [raysOK, List.all_append]

theorem flatsOK_append {rows rays : List (List Int)} {w : Nat} {a b : List Nat}
    (ha : flatsOK rows rays w a = true) (hb : flatsOK rows rays w b = true) :
    flatsOK rows rays w (a ++ b) = true := by
  simp only [flatsOK, List.all_append, Bool.and_eq_true] at *
  exact ⟨ha, hb⟩

/-- What the cone data show at a vertex with minimal vectors `mins` and rays `rays`: the rows
`symcL k` span the dual space; every ray is nonnegative on the rows; every flat is certified; and
every set of 14 rows lies in one of the flats. The parts are checked separately, the flats and
the sets of rows in pieces. -/
structure ConeOK (mins rays : List (List Int)) (cd : ConeData) : Prop where
  rank_ok : rankOK (mins.map symcL) cd.idx cd.B cd.c = true
  rays_ok : raysOK (mins.map symcL) rays = true
  flats_ok : flatsOK (mins.map symcL) rays cd.w cd.flats = true
  subs_ok : ∃ js, subsOK cd.tree 14 (List.range (mins.map symcL).length) 0 js = some []

end D4

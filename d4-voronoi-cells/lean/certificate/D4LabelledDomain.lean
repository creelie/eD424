/-
D4LabelledDomain.lean: the regions II_s and II_f of the labelled certificate
(thm:m23 of the paper; multi_cap/labelled_certificate_check.py, steps 4
and 5), in exact dyadic arithmetic.  Region I is the inequality (C) of
thm:certificate, which D4CertMain checks.

With P the certificate polynomial of D4CertDomain (scaled by 315 there), and
Q0 = 1000 (omega(u) + omega(v) + omega(t)) - P(u, v, t), both regions are
checked on the ordered admissible domain u <= v <= t,
1 + 2uvt - u^2 - v^2 - t^2 >= 0, by the branch and bound of D4CertDomain
(second-order Taylor form about the centre of each box), with the omega
tables extended to a_D = amax(D, D), D = 2.1648:

  II_s, 1/2 <= t <= t_1 = 0.51, u, v >= -1:   Q0 + c (t - 1/2) >= 0,
        c the slab constant of the data;
  II_f, t_1 <= t <= a_D, u, v >= -1:          Q0 + 1000 (Gamma_1 + Gamma_2 + Gamma_3) >= 0,
        where on a box the Gamma terms are bounded below by the envelopes of
        labelled_certificate_check.py: 256 cells of the height h in [1, D/2],
        the shares fr(tau_k, x) bounded above at 1601 inner products x in
        [1/3, a_D] (computed here, from exact square-root bounds), the cell
        integrals ds_k of the data, and the packing bound amax(2 tau_{k+1},
        2 tau_{l+1}) on the heights allowed by the lower ends of the box.

The inputs not computed here are those of D4LabelledData: the tables of
omega, omega', omega'' up to a_D, the slab constant, and the enclosures of
the cell integrals ds_k, all evaluated from closed forms in interval
arithmetic by the generator.
-/
import D4CertDomain
import D4LabelledData

/-! ## The tables -/

def usLT : Array Dy := usL.map mkDy
def olowLT : Array Dy := olowL.map mkDy
def d1loLT : Array Dy := d1loL.map mkDy
def d1hiLT : Array Dy := d1hiL.map mkDy
def duLT : Dy := mkDy duL
def m2LT : Dy := mkDy m2L
def cSlab : Dy := mkDy cSlabRaw
def aDtop : Dy := mkDy aDRaw
def t1 : Dy := mkDy t1Raw
def dsLo : Array Dy := dsLoRaw.map mkDy
def dsHi : Array Dy := dsHiRaw.map mkDy

/-- The tables of D4LabelledData, from labelled_certificate_check.py. -/
def labTab : OTab := ⟨usLT, olowLT, d1loLT, d1hiLT, duLT, m2LT⟩

/-! ## Exact rational helpers -/

def Dy.toRat (a : Dy) : Rat := (a.n : Rat) / ((2 ^ a.e : Nat) : Rat)

/-- The least multiple of 2^-p that is >= q. -/
def ratUp (q : Rat) (p : Nat) : Dy :=
  let s : Int := (q * ((2 ^ p : Nat) : Rat)).ceil
  ⟨s, p⟩

/-- A rational s >= sqrt y (y >= 0), with s^2 >= y checked; 2^-p resolution. -/
def sqrtUp (y : Rat) (p : Nat) : Rat :=
  let scale : Nat := 4 ^ p
  let n : Nat := (y * (scale : Rat)).ceil.toNat
  let r := Nat.sqrt n
  let r := if r * r ≥ n then r else r + 1
  let s : Rat := (r : Rat) / ((2 ^ p : Nat) : Rat)
  if s * s ≥ y then s else s + 1

/-! ## The constants of II_f -/

def Dd : Rat := 21648 / 10000
def Hh : Rat := Dd / 2
def KC : Nat := 256
def MX : Nat := 1600
def R2 : Rat := 3 / 2

def tauK (k : Nat) : Rat := 1 + (Hh - 1) * (k : Rat) / (KC : Rat)
def amaxR (d1 d2 : Rat) : Rat := (d1 * d1 + d2 * d2 - 4) / (2 * d1 * d2)
def xtop : Rat := aDtop.toRat
def xsM (m : Nat) : Rat := 1 / 3 + (xtop - 1 / 3) * (m : Rat) / (MX : Rat)

/-- An upper bound of fr(tau, x) = (1 - q)^2 (2 + q) / 4, q = (1 - tau x) / ((1 - x^2)^(1/2) (R^2 - tau^2)^(1/2)),
fr = 0 for q >= 1: q is bounded below through an upper bound of the square root, and
(1 - q)^2 (2 + q) decreases in q on [-1, 1]. -/
def frHi (tau x : Rat) : Rat :=
  let num := 1 - tau * x
  let den := sqrtUp ((1 - x * x) * (R2 - tau * tau)) 64
  if num ≤ 0 then 1 else
  let q := num / den
  if q ≥ 1 then 0 else (1 - q) * (1 - q) * (2 + q) / 4

/-- FR[k][0] = 0 (inner products at most 1/3), FR[k][m + 1] >= fr(tau_k, x_m), as dyadic numbers. -/
def FRtab : Array (Array Dy) :=
  (List.range KC).toArray.map fun k =>
    #[Dy.zero] ++ (List.range (MX + 1)).toArray.map fun m => ratUp (frHi (tauK k) (xsM m)) 62

/-- AMX[k][l] >= amax(2 tau_{k+1}, 2 tau_{l+1}). -/
def AMXtab : Array (Array Dy) :=
  (List.range KC).toArray.map fun k =>
    (List.range KC).toArray.map fun l => ratUp (amaxR (2 * tauK (k + 1)) (2 * tauK (l + 1))) 62

/-- The grid x_m as dyadic numbers rounded down (so that x_m >= xhi is decided safely). -/
def xsLo : Array Dy :=
  (List.range (MX + 1)).toArray.map fun m => let u := ratUp (-(xsM m)) 62; ⟨-u.n, u.e⟩

def thirdLo : Dy := let u := ratUp (-(1 / 3 : Rat)) 62; ⟨-u.n, u.e⟩

/-- Facts the lookups rely on: every row of AMX increases, the grid increases. -/
def tablesSorted : Bool :=
  AMXtab.all (fun row => (List.range (KC - 1)).all fun l => Dy.lt (row[l]!) (row[l + 1]!)) &&
  (List.range MX).all fun m => Dy.lt (xsLo[m]!) (xsLo[m + 1]!)

/-- Column of FR bounding fr(., x) for every x <= xhi: 0 if xhi <= 1/3, else m + 1 for
the least m with x_m >= xhi; none beyond the grid. -/
def colOf (xhi : Dy) : Option Nat :=
  if Dy.le xhi thirdLo then some 0 else
  let rec go (lo hi : Nat) : Nat → Nat      -- least index in [lo, hi) with xsLo >= xhi, or hi
    | 0 => hi
    | fuel + 1 =>
      if lo ≥ hi then hi else
      let mid := (lo + hi) / 2
      if Dy.le xhi (xsLo[mid]!) then go lo mid fuel else go (mid + 1) hi fuel
  let m := go 0 (MX + 1) 64
  if m ≥ MX + 1 then none else some (m + 1)

/-- env[k] <= min over h' >= (any h in cell k) of Gamma(h'), for the pair of inner products (ahi, bhi). -/
def gammaEnv (ahi bhi : Dy) : Option (Array Dy) := do
  let ca ← colOf ahi
  let cb ← colOf bhi
  let mut before : Dy := Dy.zero
  let mut g : Array Dy := #[]
  for k in [0:KC] do
    let fa := (FRtab[k]!)[ca]!
    let fb := (FRtab[k]!)[cb]!
    let cut := (dsHi[k]!).mul (fa.add fb)
    g := g.push (before.sub cut)
    -- ds_lo / 11 - cut, with 1/11 replaced by a dyadic number below it
    before := before.add (((dsLo[k]!).mul ⟨419244183493398900, 62⟩).sub cut)
  -- suffix minima
  let mut env : Array Dy := g
  for i in [0:KC - 1] do
    let k := KC - 2 - i
    env := env.set! k (Dy.min (env[k]!) (env[k + 1]!))
  return env

/-- The least l with AMX[k][l] >= ylo (KC if none), for every k. -/
def kappa (ylo : Dy) : Array Nat :=
  AMXtab.map fun row =>
    let rec go (lo hi : Nat) : Nat → Nat
      | 0 => hi
      | fuel + 1 =>
        if lo ≥ hi then hi else
        let mid := (lo + hi) / 2
        if Dy.le ylo (row[mid]!) then go lo mid fuel else go (mid + 1) hi fuel
    go 0 KC 64

/-- A lower bound of 1000 (Gamma_1 + Gamma_2 + Gamma_3) over the heights the packing allows
for the box (rows u, v, t); none if some inner product lies beyond the grid. -/
def gain (lo hi : Array Dy) : Option Dy := do
  let G1 ← gammaEnv (hi[2]!) (hi[1]!)
  let G2 ← gammaEnv (hi[2]!) (hi[0]!)
  let G3 ← gammaEnv (hi[1]!) (hi[0]!)
  let kt := kappa (lo[2]!)
  let kv := kappa (lo[1]!)
  let half : Dy := ⟨1, 1⟩
  let mut best : Option Dy := none
  if Dy.le (lo[0]!) half then
    for k1 in [0:KC] do
      let a := kt[k1]!; let b := kv[k1]!
      if a < KC && b < KC then
        let val := ((G1[k1]!).add (G2[a]!)).add (G3[b]!)
        best := some (match best with | none => val | some x => Dy.min x val)
  else
    let ku := kappa (lo[0]!)
    for k1 in [0:KC] do
      for k2 in [kt[k1]!:KC] do
        let k3 := Nat.max (kv[k1]!) (ku[k2]!)
        if k3 < KC then
          let val := ((G1[k1]!).add (G2[k2]!)).add (G3[k3]!)
          best := some (match best with | none => val | some x => Dy.min x val)
  match best with
  | none => return ⟨1000000000000, 0⟩   -- no heights are allowed: the box holds no configuration
  | some x => return x.mulInt 1000

/-! ## The branch and bound -/

inductive Extra where
  | slab (c : Dy)
  | gamma

def processL (tab : OTab) (ps : Polys) (maxAge : Nat) (extra : Extra) (b : Box) : Outcome := Id.run do
  let lo := b.lo; let hi := b.hi
  if Dy.lt (hi[1]!) (lo[0]!) || Dy.lt (hi[2]!) (lo[1]!) then return .verified
  let ivs : Array Iv := #[⟨lo[0]!, hi[0]!⟩, ⟨lo[1]!, hi[1]!⟩, ⟨lo[2]!, hi[2]!⟩]
  let pw2 := ivs.map fun x => ivPowers x 2
  let det := detPoly.evalIv (pw2[0]!) (pw2[1]!) (pw2[2]!)
  if Dy.lt det.hi Dy.zero then return .verified
  let c : Array Dy := #[(lo[0]!).add (hi[0]!) |>.half, (lo[1]!).add (hi[1]!) |>.half, (lo[2]!).add (hi[2]!) |>.half]
  let r : Array Dy := #[(hi[0]!).sub (lo[0]!) |>.half, (hi[1]!).sub (lo[1]!) |>.half, (hi[2]!).sub (lo[2]!) |>.half]
  let width := Dy.max (Dy.max ((hi[0]!).sub (lo[0]!)) ((hi[1]!).sub (lo[1]!))) ((hi[2]!).sub (lo[2]!))
  let recompute := b.hess.isNone || b.age ≥ maxAge
  let fresh : Unit → Array Dy := fun _ =>
    let pw := ivs.map fun x => ivPowers x ps.deg
    ps.H.map fun hp => (hp.evalIv (pw[0]!) (pw[1]!) (pw[2]!)).absHi
  let hess : Array Dy :=
    match b.hess with
    | some h => if recompute then fresh () else h
    | none => fresh ()
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
  -- the linear term of the slab, scaled by 315 like everything else
  match extra with
  | .slab cc =>
    let c315 := cc.mulInt 315
    q0 := q0.add (c315.mul ((c[2]!).sub ⟨1, 1⟩))
    glo := glo.set! 2 ((glo[2]!).add c315)
    ghi := ghi.set! 2 ((ghi[2]!).add c315)
  | .gamma => pure ()
  let cp := c.map fun x => dyPowers x ps.deg
  q0 := q0.sub (ps.P.evalPt (cp[0]!) (cp[1]!) (cp[2]!))
  let mut first : Dy := Dy.zero
  let mut contrib : Array Dy := #[Dy.zero, Dy.zero, Dy.zero]
  for v in [0, 1, 2] do
    let g := (ps.D[v]!).evalPt (cp[0]!) (cp[1]!) (cp[2]!)
    let ga := Dy.max ((glo[v]!).sub g).abs ((ghi[v]!).sub g).abs
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
  let mut qlo := (q0.sub first).sub second.half
  match extra with
  | .gamma =>
    if !qlo.isNonneg then
      match gain lo hi with
      | none => return .failed
      | some gl => qlo := qlo.add (gl.mulInt 315)
    -- the gain moves with the lower ends above 1/2 (a splitting heuristic, as in the script)
    for v in [0, 1, 2] do
      if Dy.lt ⟨1, 1⟩ (hi[v]!) then
        contrib := contrib.set! v ((contrib[v]!).add (((hi[v]!).sub (lo[v]!)).mulInt (130 * 315)))
  | .slab _ => pure ()
  if qlo.isNonneg then return .verified
  if Dy.lt width WMIN then return .failed
  let axis := if Dy.le (contrib[1]!) (contrib[0]!) && Dy.le (contrib[2]!) (contrib[0]!) then 0
              else if Dy.le (contrib[2]!) (contrib[1]!) then 1 else 2
  let mid := c[axis]!
  return .split ⟨lo, hi.set! axis mid, some hess, age + 1⟩ ⟨lo.set! axis mid, hi, some hess, age + 1⟩

def runL (tab : OTab) (ps : Polys) (hw : Nat) (extra : Extra) (done : Nat) : Nat → List Box → Nat × Nat
  | 0, _ => (2, done)
  | _, [] => (0, done)
  | fuel + 1, b :: rest =>
    match processL tab ps hw extra b with
    | .verified => runL tab ps hw extra (done + 1) fuel rest
    | .failed => (1, done)
    | .split b1 b2 => runL tab ps hw extra (done + 1) fuel (b1 :: b2 :: rest)

def mOne : Dy := ⟨-1, 0⟩

/-- II_s with a given table of omega: the box [-1, t1] x [-1, t1] x [1/2, t1]; (status, boxes). -/
def statSlabWith (tab : OTab) (fuel : Nat) : Nat × Nat :=
  match thePolys with
  | none => (3, 0)
  | some ps => runL tab ps HAGE (.slab cSlab) 0 fuel [⟨#[mOne, mOne, ⟨1, 1⟩], #[t1, t1, t1], none, 0⟩]

/-- II_f with a given table of omega: the box [-1, a_D] x [-1, a_D] x [t1, a_D]; (status, boxes). -/
def statGammaWith (tab : OTab) (fuel : Nat) : Nat × Nat :=
  match thePolys with
  | none => (3, 0)
  | some ps => runL tab ps HAGE .gamma 0 fuel [⟨#[mOne, mOne, t1], #[aDtop, aDtop, aDtop], none, 0⟩]

def statSlab (fuel : Nat) : Nat × Nat := statSlabWith labTab fuel
def statGamma (fuel : Nat) : Nat × Nat := statGammaWith labTab fuel

/-- The table of cosines increases, and its steps are at most duLT. -/
def usSorted : Bool :=
  (List.range (usLT.size - 1)).all fun i => Dy.lt (usLT[i]!) (usLT[i + 1]!) && Dy.le ((usLT[i + 1]!).sub (usLT[i]!)) duLT

/-- The constants: t1 >= 0.51, a_D <= aDtop, and the facts the lookups use. -/
def constantsOk : Bool :=
  usSorted &&
  (51 / 100 : Rat) ≤ t1.toRat && amaxR Dd Dd ≤ aDtop.toRat && tablesSorted &&
  ((⟨419244183493398900, 62⟩ : Dy).toRat ≤ 1 / 11) && usLT.size == olowLT.size && usLT.size == d1loLT.size &&
  usLT.size == d1hiLT.size && Dy.le aDtop (usLT[usLT.size - 1]!) && dsLo.size == KC && dsHi.size == KC


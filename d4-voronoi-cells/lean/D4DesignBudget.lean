/-
D4DesignBudget.lean: the exact content of the proposition "No room from the
design defects" (sec:closure of the paper).

With G_k = U_k/(k+1) (U_k the Chebyshev polynomials of the second kind, so
G_k(1) = 1) and p = sum_(k=0..5) p_k G_k for the rationals p_k below:

  * pmono is the list of monomial coefficients of p (checked from the U_k);
  * p(t) < 0 on [-1, th], with th = 0.614039 > tau, where
    tau^2 = 139129/369000 (as in D4HoleBudget), proved in the paper by the
    identity -p = (1 + t) A^2 + (th - t) B^2 + R: theorem two_square checks
    that R has constant term larger than the sum of the absolute values of
    its other coefficients (so R > 0 on [-1, 1]);
  * independently, the same sign by a Taylor bound about the midpoint of each
    of 29 subintervals that cover the range without gaps;
  * (24 p_0)^2 > 0.26785 * sum_(k=1..5) p_k^2 / c_k for the weights c_k;
  * over the 24 roots +-e_j +-e_k, sum over ordered pairs of G_k(<a,b>/2)
    is 0 for k = 1, ..., 5 (the root system is a spherical 5-design).

The addition formula and the Cauchy-Schwarz step of the proof are the
paper's.

Written by gen_design_budget_lean.py.  No `sorry`, no Mathlib, no native_decide.
-/

namespace D4DesignBudget

def cheb : List (List Int) := [[1], [0, 2], [-1, 0, 4], [0, -4, 0, 8], [1, 0, -12, 0, 16], [0, 6, 0, -32, 0, 32]]

def pgegen : List Rat := [(39954677 / 453230000000 : Rat), (41 / 89653 : Rat), (29 / 33806 : Rat), (17 / 14206 : Rat), (95 / 98183 : Rat), (52 / 86257 : Rat)]

def weights : List Rat := [(103 / 963 : Rat), (191 / 963 : Rat), (280 / 963 : Rat), (230 / 963 : Rat), (53 / 321 : Rat)]

def pmono : List Rat := [(-9643697013659281 / 2256524186592810000000 : Rat), (-14996588399 / 109857822451126 : Rat), (-5867038 / 4978761747 : Rat), (-1510589 / 1838050413 : Rat), (304 / 98183 : Rat), (832 / 258771 : Rat)]

def th : Rat := (614039 / 1000000 : Rat)

def sqA : List Rat := [(26327 / 100000000 : Rat), (14931 / 25000000 : Rat), (3523 / 3125000 : Rat)]

def sqB : List Rat := [(260109 / 100000000 : Rat), (4473253 / 100000000 : Rat), (177231 / 3125000 : Rat)]

/-- coefficient i of sum_k g_k U_k/(k+1) -/
def monoOf (g : List Rat) (i : Nat) : Rat :=
  (List.range g.length).foldl (fun s k =>
    s + g.getD k 0 * ((cheb.getD k []).getD i 0 : Int) / ((k + 1 : Nat) : Rat)) 0

theorem pmono_ok :
    (pmono.length == 6 && (List.range 6).all (fun i => monoOf pgegen i == pmono.getD i 0)) = true := by
  decide +kernel

/-- one pass of synthetic division by (t - m) from the top down to index j -/
def shiftPass (m : Rat) (j : Nat) (c : List Rat) : List Rat :=
  (List.range (c.length - 1 - j)).foldl (fun c t =>
    let i := c.length - 2 - t
    c.set i (c.getD i 0 + m * c.getD (i + 1) 0)) c

/-- the coefficients of cs(m + d) as a polynomial in d -/
def shift (cs : List Rat) (m : Rat) : List Rat :=
  (List.range (cs.length - 1)).foldl (fun c j => shiftPass m j c) cs

def rabs (x : Rat) : Rat := if x < 0 then -x else x

/-- the product of two polynomials, constant term first -/
def pmul (a b : List Rat) : List Rat :=
  (List.range (a.length + b.length - 1)).map (fun n =>
    (List.range (n + 1)).foldl (fun s i => s + a.getD i 0 * b.getD (n - i) 0) 0)

/-- the sum of two polynomials, constant term first -/
def padd (a b : List Rat) : List Rat :=
  (List.range (max a.length b.length)).map (fun i => a.getD i 0 + b.getD i 0)

/-- R = -p - (1 + t) A^2 - (th - t) B^2 -/
def rest : List Rat :=
  padd (pmono.map (fun x => -x))
    ((padd (pmul [1, 1] (pmul sqA sqA)) (pmul [th, -1] (pmul sqB sqB))).map (fun x => -x))

/-- R > 0 on [-1, 1], because its constant term beats the other coefficients;
so -p = (1 + t) A^2 + (th - t) B^2 + R > 0 on [-1, th] -/
theorem two_square :
    (rest.length == 6 &&
      decide (rest.tail.foldl (fun s x => s + rabs x) 0 < rest.getD 0 0)) = true := by
  decide +kernel

/-- an upper bound for the polynomial cs on [a, b]: c_0 + sum |c_i| r^i about the midpoint -/
def upper (cs : List Rat) (a b : Rat) : Rat :=
  let m := (a + b) / 2
  let r := (b - a) / 2
  let c := shift cs m
  (c.tail.foldl (fun (acc : Rat × Rat) x => (acc.1 + rabs x * (acc.2 * r), acc.2 * r)) (c.getD 0 0, 1)).1

/-- the intervals cover [lo, hi] from left to right without gaps -/
def covers : List (Rat × Rat) -> Rat -> Rat -> Bool
  | [], lo, hi => lo == hi
  | (a, b) :: rest, lo, hi => a == lo && a < b && covers rest b hi

def cover1 : List (Rat × Rat) := [
  ((-1 : Rat), (-14385961 / 16000000 : Rat)),
  ((-14385961 / 16000000 : Rat), (-27157883 / 32000000 : Rat)),
  ((-27157883 / 32000000 : Rat), (-6385961 / 8000000 : Rat)),
  ((-6385961 / 8000000 : Rat), (-49473649 / 64000000 : Rat)),
  ((-49473649 / 64000000 : Rat), (-4785961 / 6400000 : Rat)),
  ((-4785961 / 6400000 : Rat), (-94105181 / 128000000 : Rat)),
  ((-94105181 / 128000000 : Rat), (-186596323 / 256000000 : Rat)),
  ((-186596323 / 256000000 : Rat), (-46245571 / 64000000 : Rat)),
  ((-46245571 / 64000000 : Rat), (-36673649 / 51200000 : Rat)),
  ((-36673649 / 51200000 : Rat), (-90877103 / 128000000 : Rat)),
  ((-90877103 / 128000000 : Rat), (-11157883 / 16000000 : Rat)),
  ((-11157883 / 16000000 : Rat), (-20701727 / 32000000 : Rat)),
  ((-20701727 / 32000000 : Rat), (-2385961 / 4000000 : Rat)),
  ((-2385961 / 4000000 : Rat), (-3157883 / 8000000 : Rat)),
  ((-3157883 / 8000000 : Rat), (-385961 / 2000000 : Rat)),
  ((-385961 / 2000000 : Rat), (-4561337 / 32000000 : Rat)),
  ((-4561337 / 32000000 : Rat), (-1501727 / 12800000 : Rat)),
  ((-1501727 / 12800000 : Rat), (-1473649 / 16000000 : Rat)),
  ((-1473649 / 16000000 : Rat), (-10175153 / 128000000 : Rat)),
  ((-10175153 / 128000000 : Rat), (-4280557 / 64000000 : Rat)),
  ((-4280557 / 64000000 : Rat), (-277883 / 5120000 : Rat)),
  ((-277883 / 5120000 : Rat), (-1333259 / 32000000 : Rat)),
  ((-1333259 / 32000000 : Rat), (-1052479 / 64000000 : Rat)),
  ((-1052479 / 64000000 : Rat), (14039 / 1600000 : Rat)),
  ((14039 / 1600000 : Rat), (1894819 / 32000000 : Rat)),
  ((1894819 / 32000000 : Rat), (1754429 / 16000000 : Rat)),
  ((1754429 / 16000000 : Rat), (842117 / 4000000 : Rat)),
  ((842117 / 4000000 : Rat), (3298273 / 8000000 : Rat)),
  ((3298273 / 8000000 : Rat), (614039 / 1000000 : Rat))]

theorem p_negative_below_th :
    (covers cover1 (-1) th && cover1.all (fun ab => upper pmono ab.1 ab.2 < 0)) = true := by
  decide +kernel

theorem th_above_tau : (0 < th && 139129 / 369000 < th * th) = true := by
  decide +kernel

theorem threshold :
    (26785 / 100000 * ((List.range 5).foldl (fun s k =>
        s + pgegen.getD (k + 1) 0 * pgegen.getD (k + 1) 0 / weights.getD k 1) 0) <
      (24 * pgegen.getD 0 0) * (24 * pgegen.getD 0 0)) = true := by
  decide +kernel

def roots : List (List Int) := [[1, 1, 0, 0],
  [1, -1, 0, 0],
  [-1, 1, 0, 0],
  [-1, -1, 0, 0],
  [1, 0, 1, 0],
  [1, 0, -1, 0],
  [-1, 0, 1, 0],
  [-1, 0, -1, 0],
  [1, 0, 0, 1],
  [1, 0, 0, -1],
  [-1, 0, 0, 1],
  [-1, 0, 0, -1],
  [0, 1, 1, 0],
  [0, 1, -1, 0],
  [0, -1, 1, 0],
  [0, -1, -1, 0],
  [0, 1, 0, 1],
  [0, 1, 0, -1],
  [0, -1, 0, 1],
  [0, -1, 0, -1],
  [0, 0, 1, 1],
  [0, 0, 1, -1],
  [0, 0, -1, 1],
  [0, 0, -1, -1]]

def idot (a b : List Int) : Int := (List.zipWith (fun x y => x * y) a b).foldl (fun s x => s + x) 0

/-- G_k(x) = U_k(x)/(k+1) -/
def gk (k : Nat) (x : Rat) : Rat :=
  (((cheb.getD k []).map (fun (c : Int) => (c : Rat))).foldr (fun cf s => cf + x * s) 0) / ((k + 1 : Nat) : Rat)

theorem roots_design :
    (roots.length == 24 && (List.range 5).all (fun k =>
      roots.foldl (fun s a => roots.foldl (fun s b => s + gk (k + 1) ((idot a b : Rat) / 2)) s) 0 == 0)) = true := by
  decide +kernel

end D4DesignBudget

#print axioms D4DesignBudget.two_square
#print axioms D4DesignBudget.p_negative_below_th
#print axioms D4DesignBudget.threshold
#print axioms D4DesignBudget.roots_design

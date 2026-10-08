/-
D4OmegaMain.lean: the checks of D4LabelledMain again, with nothing taken from
outside Lean but the certificate: the tables of omega are computed in D4Omega
from the closed forms, the slab constant and the enclosures of the cell integrals
of the data are checked against values computed there, and so is the comparison
of A_* with the bound of the certificate of thm:certificate.  Settled by
native_decide.  D4OmegaRegionI.lean does the same for region I (thm:certificate).
-/
import D4Omega

/-- A_* > 8 - 0.0928555703 (the bound of the certificate, D4Certificate.lean); s(D) > 8 - A_*;
a_D / (1 - (D/2) a_D) < 2; fr(1, 1/2) < 1/22; the slab constant of the data at most 4000 r kappa; and the 256 cell
integrals of the data enclosing the values computed here. -/
theorem omega_constants : labelledConstantsOk = true := by native_decide

/-- Corollary "at most twenty-two centres within sqrt 6": (9/8) pi^2 - 22 S(2) > 8.046. -/
theorem count_bound :
    ((RI.sub (RI.scaleRat (9/8) (RI.mul piI piI)) (RI.scaleRat 22 (capS (RI.ofNat 1)))).loRat > 8046 / 1000) = true := by
  native_decide

/-- A rational c >= q^(1/3) (q >= 0), from the integer cube root of q * 2^192. -/
def cbrtUp (q : Rat) : Rat :=
  let n : Nat := (q * ((2 ^ 192 : Nat) : Rat)).ceil.toNat
  -- the least r with r^3 >= n, by bisection
  let rec go (lo hi : Nat) : Nat → Nat
    | 0 => hi
    | fuel + 1 => if lo + 1 ≥ hi then hi else
        let mid := (lo + hi) / 2
        if mid * mid * mid ≥ n then go lo mid fuel else go mid hi fuel
  ((go 0 (2 ^ 80) 200 : Nat) : Rat) / ((2 ^ 64 : Nat) : Rat)

/-- The covering bound (pi m / 3) tan^3 r_m, 2 r_m - sin 2 r_m = 2 pi / m, exceeds 8: with
rho = arctan c, c^3 = 24 / (pi m), that is 2 rho - sin 2 rho < 2 pi / m, since 2r - sin 2r
increases; and 2 rho - sin 2 rho = 2 arctan c - 2c / (1 + c^2) increases in c, so it is
enough at a rational c above the cube root. -/
def coveringOk (m : Nat) : Bool :=
  let c := cbrtUp (24 / (piI.loRat * m))
  let lhs := RI.sub (RI.scaleRat 2 (atanI (RI.ofRat c))) (RI.ofRat (2 * c / (1 + c * c)))
  let rhs := RI.div (RI.scaleRat 2 piI) (RI.ofNat m)
  Dy.lt lhs.hi rhs.lo

/-- The covering bound exceeds 8 for every m from 5 to 22. -/
theorem covering_above_8 : ((List.range 18).all fun i => coveringOk (i + 5)) = true := by native_decide

/-- The two tables: increasing grids of step 2^-16 up to 1/2 and up to a_D, a positive
bound on |omega''|, and d1lo <= d1hi throughout. -/
theorem omega_tables : (tableOk leanTabI ⟨1, 1⟩ && tableOk leanTabL aDtop) = true := by native_decide

def verifySlabLean : Bool := (statSlabWith leanTabL 20000000).1 == 0
def verifyGammaLean : Bool := (statGammaWith leanTabL 20000000).1 == 0

/-- II_s with the tables of omega computed in Lean. -/
theorem region_IIs_lean : verifySlabLean = true := by native_decide

/-- II_f with the tables of omega computed in Lean. -/
theorem region_IIf_lean : verifyGammaLean = true := by native_decide

#eval (leanTabI.us.size, leanTabL.us.size, leanTabI.m2.toRat.num.toFloat / leanTabI.m2.toRat.den.toFloat,
       leanTabL.m2.toRat.num.toFloat / leanTabL.m2.toRat.den.toFloat)
#eval (statSlabWith leanTabL 20000000, statGammaWith leanTabL 20000000)
#print axioms count_bound
#print axioms covering_above_8
#print axioms region_IIf_lean

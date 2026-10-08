/-
D4LabelledMain.lean: the regions II_s and II_f of the labelled certificate
(thm:m23), with region I in D4CertMain.  Each theorem says that a check
of D4LabelledDomain returns true; they are settled by native_decide, which
compiles and runs the check, and so trusts the Lean compiler as well as the
kernel.
-/
import D4LabelledDomain

/-- The two runs, with a fuel of twenty million boxes each (defined here, not in
D4LabelledDomain, for the reason given in D4CertMain). -/
def verifySlab : Bool := (statSlab 20000000).1 == 0
def verifyGamma : Bool := (statGamma 20000000).1 == 0

/-- t_1 >= 0.51, a_D <= the top of the boxes, 1/11 bounded below by the dyadic number used,
the tables sorted, with steps at most du, and reaching a_D. -/
theorem labelled_constants : constantsOk = true := by native_decide

/-- II_s: Q0 + c (t - 1/2) >= 0 on the ordered admissible domain with 1/2 <= t <= t_1. -/
theorem region_IIs : verifySlab = true := by native_decide

/-- II_f: Q0 + 1000 (Gamma_1 + Gamma_2 + Gamma_3) >= 0 on the ordered admissible domain
with t_1 <= t <= a_D, the Gamma terms bounded below over the heights the packing allows. -/
theorem region_IIf : verifyGamma = true := by native_decide

#print axioms region_IIf

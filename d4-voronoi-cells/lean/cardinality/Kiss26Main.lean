/-
Kiss26Main.lean: the certificate of degree 10 at t = 0.51468 (thm:twenty-six),
checked in Lean with the routines of KissDomain.  Each theorem says that a
check returns true; they are settled by native_decide, which compiles and runs
the check, and so trusts the Lean compiler as well as the kernel.  With the
Schoenberg and Bachoc-Vallentin positivity that the paper cites (not
formalised here), they give: every set of points of S^3 with pairwise inner
products at most 1/2 + 0.01468 has at most 25 elements.
-/
import Kiss26Data
import KissChecks

def checked26 : Checked := prepare cert26

/-- (|C| - 1)(1 - e1) - (|C| - 1)(|C| - 2) e2 > B - 1 at |C| = m, with the decimals of the paper. -/
def Cert.countOkAt (c : Cert) (F : RPoly) (m : Nat) : Bool :=
  ((m : Rat) - 1) * (1 - c.e1Dec) - ((m : Rat) - 1) * ((m : Rat) - 2) * c.e2Dec > c.bound F - 1

/-- The checks of Checked.exactOk, with the count taken at |C| = m. -/
def Checked.exactOkAt (k : Checked) (c : Cert) (m : Nat) : Bool :=
  c.positivityOk && c.thresholdsOk && c.countOkAt k.F m && k.F.symmetric

/-- f_k >= 0; F_k symmetric positive definite; top >= t; e1 <= 10^-6, e2 <= 2.9 10^-4;
25 (1 - e1) - 600 e2 > B - 1 with B = 1 + f(1) + F(1,1,1) exact; F symmetric. -/
theorem cert26_exact : checked26.exactOkAt cert26 26 = true := by native_decide

/-- (i): f(u) + 3 F(1,u,u) + 1 <= e1 for -1 <= u <= top. -/
theorem cert26_i : checked26.okI = true := by native_decide

/-- (ii): F(u,v,w) <= e2 for -1 <= u <= v <= w <= top, 1 + 2uvw - u^2 - v^2 - w^2 >= 0. -/
theorem cert26_ii : checked26.okII = true := by native_decide

#print axioms cert26_ii

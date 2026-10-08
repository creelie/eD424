/-
KissMain8.lean: the certificate of degree 8 at s = 0.0065, checked in Lean.
Each theorem says that a check of KissDomain returns true; they are settled
by native_decide, which compiles and runs the check, and so trusts the Lean
compiler as well as the kernel.  With the Schoenberg and Bachoc-Vallentin
positivity that the paper cites (not formalised here), they give: every set
of points of S^3 with pairwise inner products at most 1/2 + 0.0065 has at most
24 elements.
-/
import KissChecks

/-- f_k >= 0; F_k symmetric positive definite; top >= t; e1 <= 10^-6, e2 <= e2Dec;
24 (1 - e1) - 552 e2 > B - 1 with B = 1 + f(1) + F(1,1,1) exact; F symmetric. -/
theorem cert8_exact : checked8.exactOk cert8 = true := by native_decide

/-- (i): f(u) + 3 F(1,u,u) + 1 <= e1 for -1 <= u <= top. -/
theorem cert8_i : checked8.okI = true := by native_decide

/-- (ii): F(u,v,w) <= e2 for -1 <= u <= v <= w <= top, 1 + 2uvw - u^2 - v^2 - w^2 >= 0. -/
theorem cert8_ii : checked8.okII = true := by native_decide

#print axioms cert8_ii

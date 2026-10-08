/-
Count31Main.lean: the certificate of thm:count31 checked in Lean with the routines
of CountDomain.  Each theorem says that a check returns true; they are settled by
native_decide, which compiles and runs the check, and so trusts the Lean compiler
as well as the kernel.  With the positivity of the zonal kernels and the
two-point certificate theorem of the paper (not formalised here), they give:
every packing set of at least 31 centres within sqrt 6 of a centre has
U(Y) <= 31 m + t/2 < 9 pi^2/8 - 8, so T(Y) > 8.
-/
import CountData31

/-- Check 1: A_1, ..., A_12 positive semidefinite and A_0 positive definite by exact
LDL^T, and [[A_0, z], [z^T, t]] positive semidefinite with t = z^T A_0^{-1} z
rounded up to a multiple of 2^-48. -/
theorem count31_positivity : cert31.positivityOk = true := by native_decide

/-- Check 2: K(d, d', u) <= Pi(d/2, d'/2, u) for 2 <= d <= d' <= dmax and
-1 <= u <= a(d, d'), with dmax^2 > 6. -/
theorem count31_pairs : cert31.pairsOk = true := by native_decide

/-- Check 3: S(d) + K(d, d, 1)/2 - z . p(d) <= m on [2, dmax]. -/
theorem count31_bracket : cert31.bracketOk = true := by native_decide

/-- Check 4: m < 0 and 31 m + t/2 < 9 pi^2/8 - 8. -/
theorem count31_final : cert31.finalOk = true := by native_decide

#print axioms count31_pairs

/-
D4CertMain.lean: the statement.  `verifyDomain` runs the check of
D4CertDomain, which expands the certificate polynomial exactly, scales it
by 315, and runs the interval branch and bound in exact dyadic arithmetic
over the ordered admissible domain, with the tables of D4CertData for the
omega terms.  The theorem
says that it returns true; it is settled by native_decide, which compiles
and runs the check, and so trusts the Lean compiler as well as the kernel.
-/
import D4CertDomain

/-- The whole check with a fuel of four million boxes.  It is defined here and
not in D4CertDomain because a closed definition of a precompiled module is
evaluated when the module is loaded, and D4CertDomain is imported elsewhere. -/
def verifyDomain : Bool := verifyWithFuel 4000000

theorem domain_ok : verifyDomain = true := by native_decide

#eval monomialCount
#print axioms domain_ok

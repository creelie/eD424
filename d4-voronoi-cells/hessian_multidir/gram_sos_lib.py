"""
gram_sos_lib.py
================
Small, self-contained library implementing the standard "Gram matrix"
method for testing whether a quartic form in n variables admits a
sum-of-squares (SOS) certificate, via semidefinite programming (cvxpy).

Method (standard, not novel): let m(x) be the vector of all degree-2
monomials in x (length C(n+1,2)). Any quartic form Q can be written as
Q(x) = m(x)^T M m(x) for some symmetric matrix M, but this representation
is NOT unique (different M can give the same Q, because distinct monomial
products can coincide, e.g. x0^2*x1^2 = (x0x1)*(x0x1)). The set of valid M
for a given Q is an affine subspace. Q is SOS iff some M in that affine
subspace is positive semidefinite (M>=0 psd => Q = m^T M m = ||sqrt(M) m||^2,
an explicit sum of squares; conversely a PSD M in the affine subspace,
if one exists, gives a certificate). This is exactly the tool Hilbert's
classification concerns: for n=4, degree=4 (quaternary quartics), this
affine subspace can be entirely free of PSD matrices even when Q is
non-negative everywhere -- the Choi-Lam polynomial below is the standard
worked example, used here purely to validate this code is implemented
correctly, not as a claim about the D4 problem.
"""
import itertools
import numpy as np
import cvxpy as cp


def deg2_monomials(n):
    """All exponent tuples of total degree 2 in n variables."""
    mons = []
    for i in range(n):
        e = [0] * n
        e[i] = 2
        mons.append(tuple(e))
    for i in range(n):
        for j in range(i + 1, n):
            e = [0] * n
            e[i] = 1
            e[j] = 1
            mons.append(tuple(e))
    return mons


def deg4_monomials(n):
    """All exponent tuples of total degree 4 in n variables, sorted."""
    mons = set()
    for combo in itertools.combinations_with_replacement(range(n), 4):
        e = [0] * n
        for c in combo:
            e[c] += 1
        mons.add(tuple(e))
    return sorted(mons)


def build_gram_map(n=4):
    """
    Returns (mons2, mons4, pairs_for) where:
      mons2: list of degree-2 monomial exponent tuples, length k = C(n+1,2)
      mons4: list of degree-4 monomial exponent tuples, length C(n+3,4)... (35 for n=4)
      pairs_for: dict mapping a degree-4 monomial tuple -> list of (i,j) index
                 pairs into mons2 (i<=j) whose product equals that monomial.
    """
    mons2 = deg2_monomials(n)
    mons4 = deg4_monomials(n)
    k = len(mons2)
    pairs_for = {m4: [] for m4 in mons4}
    for i in range(k):
        for j in range(i, k):
            e = tuple(a + b for a, b in zip(mons2[i], mons2[j]))
            pairs_for[e].append((i, j))
    return mons2, mons4, pairs_for


def fit_gram_and_check_sos(coeffs, n=4, verbose=True):
    """
    coeffs: dict mapping degree-4 exponent tuple (n-tuple summing to 4) ->
            coefficient of that monomial in Q(x).
    Returns (status, max_t, M_value) where status is 'SOS_FOUND',
    'NOT_FOUND' (SDP could not certify PSD membership of the affine
    family), or 'SOLVER_ERROR'. max_t is the optimal objective of
        maximize t   s.t.  M - t*I >> 0,  M in affine family matching coeffs
    (t>0 means a strictly PSD, hence genuinely SOS, matrix was found;
    t<=0 means the solver could not certify SOS -- NOT the same as a proof
    of non-existence, since SDP infeasibility certificates from a numerical
    solver are themselves floating point, but a strongly informative
    negative signal, exactly mirroring the textbook Choi-Lam situation
    validated below).
    """
    mons2, mons4, pairs_for = build_gram_map(n)
    k = len(mons2)
    M = cp.Variable((k, k), symmetric=True)
    t = cp.Variable()

    constraints = [M - t * np.eye(k) >> 0]
    for m4 in mons4:
        target = coeffs.get(m4, 0.0)
        terms = []
        for (i, j) in pairs_for[m4]:
            mult = 1 if i == j else 2
            terms.append(mult * M[i, j])
        constraints.append(cp.sum(terms) == target)

    prob = cp.Problem(cp.Maximize(t), constraints)
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception as e:
        if verbose:
            print("solver error:", e)
        return "SOLVER_ERROR", None, None

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return "SOLVER_ERROR_STATUS_%s" % prob.status, None, None

    max_t = t.value
    status = "SOS_FOUND" if max_t is not None and max_t > 1e-7 else "NOT_FOUND"
    return status, max_t, (M.value if M.value is not None else None)


def _validate():
    """Sanity checks against two textbook quartics, n=4:
       (a) (x0^2+x1^2+x2^2+x3^2)^2  -- a perfect square, trivially SOS.
       (b) Choi-Lam: x0^2 x1^2 + x1^2 x2^2 + x2^2 x0^2 + x3^4 - 4 x0 x1 x2 x3
           -- classically non-negative (AM-GM) but PROVABLY NOT SOS
           (Choi & Lam, 1977). This is the standard example demonstrating
           exactly the Hilbert-classification gap the paper's Section
           "Why quartic positivity is not merely a harder version of the
           same problem" describes; used here only to confirm this SDP
           code correctly distinguishes the SOS and non-SOS cases before
           trusting it on any real data.
    """
    import sympy as sp
    xs = sp.symbols('x0 x1 x2 x3')

    def coeffs_of(expr, n=4):
        expr = sp.expand(expr)
        poly = sp.Poly(expr, *xs)
        out = {}
        for monom, coeff in poly.terms():
            out[monom] = float(coeff)
        return out

    print("Validation (a): (x0^2+x1^2+x2^2+x3^2)^2  -- must find SOS")
    q1 = (xs[0]**2 + xs[1]**2 + xs[2]**2 + xs[3]**2) ** 2
    c1 = coeffs_of(q1)
    status1, t1, _ = fit_gram_and_check_sos(c1)
    print("  status:", status1, " max_t:", t1)
    assert status1 == "SOS_FOUND", "FAILED: known-SOS case not detected as SOS"

    print()
    print("Validation (b): Choi-Lam polynomial -- must NOT find SOS")
    q2 = xs[0]**2*xs[1]**2 + xs[1]**2*xs[2]**2 + xs[2]**2*xs[0]**2 + xs[3]**4 - 4*xs[0]*xs[1]*xs[2]*xs[3]
    c2 = coeffs_of(q2)
    status2, t2, _ = fit_gram_and_check_sos(c2)
    print("  status:", status2, " max_t:", t2)
    assert status2 == "NOT_FOUND", "FAILED: known-non-SOS case incorrectly found SOS"

    print()
    print("Both validations passed: this SDP code correctly separates a")
    print("known-SOS quartic from a known nonnegative-but-non-SOS quartic.")


if __name__ == "__main__":
    _validate()

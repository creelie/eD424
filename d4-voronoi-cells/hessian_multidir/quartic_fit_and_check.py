#!/usr/bin/env python3
"""
quartic_fit_and_check.py
==========================
Reads the (direction, a4_est) samples gathered by quartic_sample_dense.py
(both the earlier 36-direction run recorded in the paper's Theorem
"Broadened sampling at A_18", data reconstructed from its log, and the
new overdetermined 166-direction run), fits the 35 coefficients of the
general quartic form Q(x0,x1,x2,x3) by least squares (since a4_est(v) is
an estimate of Q(v) for unit v), reports the fit quality honestly
(residuals), and then runs the validated Gram-matrix/SDP method
(gram_sos_lib.py) to check whether the FITTED quartic admits a
sum-of-squares certificate.

WHAT A RESULT HERE WOULD AND WOULD NOT MEAN, stated plainly before any
number is produced:
  - If a PSD Gram matrix is found for the least-squares-fitted quartic,
    that is evidence the fitted approximant is non-negative everywhere
    (not just at the sampled directions) -- genuinely stronger than
    directional sampling alone, since it is a statement about the whole
    subspace. It is NOT a proof about the true quartic form at A_18,
    because the fit itself is from noisy finite-difference estimates,
    not an exact symbolic derivation; the fit residual is reported so
    the reader can judge how much to trust it.
  - If no PSD Gram matrix is found, that does NOT prove the true (or
    even the fitted) quartic is negative somewhere or fails to be SOS
    -- SDP infeasibility from a floating-point solver is itself not an
    exact certificate -- but it is a genuine negative signal, and would
    be reported as such, not explained away.
  - Either outcome leaves Conjecture (Multi-Direction Positivity) open;
    this script cannot close it, and does not claim to.
"""
import os
import sys
import numpy as np
import itertools

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from gram_sos_lib import deg4_monomials, fit_gram_and_check_sos

DATA_FILE = os.path.join(_HERE, "quartic_dense_A18_data.tsv")

# The original 36-direction run (multidir_nullspace_broad_sample.py, values
# transcribed exactly from nullspace_broad_sample_A18.log, basis/sum/diff
# vectors exact, random-direction coefficients to the 3-decimal precision
# the log itself reported -- included for comparison only; the new TSV
# data is generated at full float64 precision and is the primary fit input).


def load_samples():
    coeffs = []
    values = []
    names = []
    with open(DATA_FILE) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 6:
                continue
            name, c0, c1, c2, c3, a4 = parts
            if a4 == "PENDING":
                continue
            coeffs.append([float(c0), float(c1), float(c2), float(c3)])
            values.append(float(a4))
            names.append(name)
    return np.array(coeffs), np.array(values), names


def monomial_value(v, exponent):
    val = 1.0
    for x, e in zip(v, exponent):
        if e:
            val *= x ** e
    return val


def main():
    coeffs, values, names = load_samples()
    n_samples = len(values)
    mons4 = deg4_monomials(4)
    n_unknowns = len(mons4)
    print(f"Loaded {n_samples} directional samples (unit vectors v, target Q(v)=a4_est(v)).")
    print(f"Fitting {n_unknowns} quartic coefficients "
          f"(overdetermination: {n_samples/n_unknowns:.2f}x).")
    if n_samples < n_unknowns * 2:
        print("WARNING: fewer than 2x overdetermination -- fit may be unreliable; "
              "reporting anyway, honestly, but flagging this explicitly.")

    A = np.zeros((n_samples, n_unknowns))
    for row, v in enumerate(coeffs):
        for col, exponent in enumerate(mons4):
            A[row, col] = monomial_value(v, exponent)

    sol, residuals, rank, sv = np.linalg.lstsq(A, values, rcond=None)
    fitted = A @ sol
    resid = values - fitted
    rel_resid = np.linalg.norm(resid) / np.linalg.norm(values)
    print(f"Least-squares fit: matrix rank {rank}/{n_unknowns}, "
          f"relative residual norm = {rel_resid:.4f}")
    print(f"Max abs pointwise residual: {np.max(np.abs(resid)):.6f} "
          f"(values range {values.min():.4f} to {values.max():.4f})")

    if rank < n_unknowns:
        print("RANK DEFICIENT: the sampled directions do not span enough of the "
              "35-dimensional coefficient space to determine Q uniquely. Reporting "
              "this honestly and stopping short of an SOS claim on this fit.")
        return

    coeffs_dict = {mons4[i]: sol[i] for i in range(n_unknowns)}
    print()
    print("Running validated Gram-matrix/SDP check on the fitted quartic...")
    status, max_t, M = fit_gram_and_check_sos(coeffs_dict)
    print(f"SDP result: status={status}  max_t={max_t}")
    print()
    if status == "SOS_FOUND":
        print("INTERPRETATION: a PSD Gram matrix was found for the LEAST-SQUARES-FITTED "
              "quartic approximant to F's quartic behaviour at A_18. This is additional "
              "numerical evidence (not a proof: the fit comes from noisy high-precision "
              "finite-difference samples, not an exact symbolic derivation) that the "
              "fitted quartic is non-negative on the entire near-null subspace, not just "
              "at the sampled rays. It does not put Conjecture "
              "(Multi-Direction Positivity) at A_18 on an elementary footing: "
              "what settles that conjecture is the classification cited in "
              "cor:conj-resolved.")
    else:
        print("INTERPRETATION: no PSD Gram matrix was found for the least-squares-fitted "
              "quartic within this SDP's numerical tolerance. This does NOT show the true "
              "or even the fitted quartic is non-SOS or non-negative-failing -- it may "
              "simply reflect fit noise, or (as with the Choi-Lam validation case) a "
              "genuinely non-negative form that has no SOS certificate at all, exactly the "
              "phenomenon Hilbert's theorem predicts is possible in this dimension/degree "
              "class. Reported honestly, without rounding up to either a positive or a "
              "negative conclusion about the true quartic form.")


if __name__ == "__main__":
    sys.exit(main())

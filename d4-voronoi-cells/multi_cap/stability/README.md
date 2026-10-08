# Stability near the root system

Scripts behind the corollary "No room beside a near root system" and the
remarks on what (C) needs.  Run with Python 3, numpy and scipy, from this
directory.

Exact (rational-interval) checks, repeated in `lean/D4HoleBudget.lean`:

- `hole_budget.py`: part (ii), the hole budget.  For every unit z the sum
  over the 24 normalised roots of (<z, r> - tau)_+^2 is at least 6h^2, with
  tau = a(sqrt6, rho), rho = 2/sqrt(1 - 0.016) and h = 1/sqrt2 - tau
  (quartic minorant and the 5-design identities, checked in rationals, plus
  a random numerical check).
- `gram_to_rotation.py`: part (iii).  The identity D^2 = alpha^2 + 12e, the
  bound on the Procrustes distance, a numerical check of both on 20 000
  configurations, and the threshold 0.3059 in rational arithmetic.

Floating-point searches (evidence, not bounds):

- `maxfrob.py`, `maxfrob23.py`: the largest Procrustes distance from the
  root system (or the root system minus one root) over 24-point (23-point)
  codes of slack 0.008 found by local maximisation; best codes in
  `maxfrob_best_s0.008.npy` and `maxfrob23_best_s0.008.npy`.
- `check_gram_bound.py`, `fstar.py`: the Gram-to-rotation comparison and the
  hole budget, numerically, on the codes found.
- `maxdist.py`: the largest sum of squared distances of the inner products
  from {-1, -1/2, 0, 1/2} over k-point codes of slack 0.008
  (`maxdist_s0.008_*.json`).
- `code23.py`, `polish23.py`: 23-point codes of smallest largest inner
  product; besides the root system minus one root, a second family with
  largest inner product 0.50810886 (`code23_best.npy`,
  `code23_fam2_*.npy`).
- `hole23.py`: two further centres near sqrt6 beside the root system minus
  one root need a root-sum-square displacement of only about 0.132.

Logs of the runs quoted in the paper are in `logs/`.

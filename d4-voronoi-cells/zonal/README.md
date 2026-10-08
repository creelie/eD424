# Re-verifying the two remaining steps of the LLM24 certificate

This directory re-runs, on an ordinary machine, the two steps of the
verification of

> D. de Laat, N. M. Leijenhorst, W. H. H. de Muinck Keizer,
> *Optimality and uniqueness of the D4 root system*, arXiv:2404.18794,
> data: 4TU.ResearchData, doi:10.4121/74ce1c25-6fca-4680-8a36-e9c18e7e9594
> (`LasserreSphericalCodes.zip`, MD5 `02acd5270f7b3fa799abdeb5291706fd`)

that `llm24_certificate_check.py` had to leave out: step 3, the construction of
the zonal matrices `Z_lambda`, and step 5, the check that the four constraint
polynomials of the certificate vanish identically.  The authors' own
implementation of step 3 needs about three days and 128 GB of memory.  The code
here does the same computation in about two hours on two cores in under 400 MB,
and the whole of steps 3 to 5 in about five, in exact rational arithmetic
throughout.

Nothing here is a translation of the authors' Julia package.  It is written from
the mathematical description in their paper and the file formats in their
README, and it is checked against independent facts (below) rather than against
their output.

## Why the memory is not needed

For a signature `lambda` of `O(4)` with `|lambda| <= 14` and admissible indices
`k1, k2`, step 3 wants

```
P(S)_{k1,k2} = int_{O(4)} rho_{0,k1}(omega gamma eps) rho_{0,k2}(omega gamma S) dgamma,
```

a polynomial in the entries of `S`.  The integral is taken monomial by monomial
in the sixteen entries of `gamma`, so the authors expand the product of the two
representation polynomials first and hold it in a dictionary keyed by the
exponent matrix of `gamma`, with a polynomial in `S` for each key.  That
dictionary is the 128 GB.

It never has to exist.  The output `P(S)` is homogeneous of degree `|lambda|` in
seven variables, so it has at most `C(20,6) = 38760` coefficients whatever
`lambda` is.  Both factors are products of powers of single entries,

```
rho_A = det(A)^l2 A11^(m-k1) A12^k1,
rho_B = B11^(l2+m-k2) B22^l2 B12^k2,      m = l1 - l2,
```

and the second splits as `U * V` with `U` carrying rows 0 and 2 of `gamma` and
`V` rows 1 and 3.  Walking the triples of monomials `(a, u, v)` and adding each
one straight into the coefficient of `P(S)` it belongs to needs only the three
factors (at most 312060 terms), the accumulator, and a memo of the distinct
canonical exponent matrices seen so far (at most 2505430).

Two things make the walk cheap enough.  The integral vanishes unless every row
sum and every column sum of the exponent matrix is even, so the triples are
visited parity class by parity class: 7.854e9 of the 2.392e11 triples survive,
a factor of about 30.  And the integral is invariant under permuting rows and
permuting columns, so it is memoised on the canonical form, which collapses the
distinct matrices by two or three orders of magnitude.

## The files

| file | what it does |
| --- | --- |
| `o4.py` | the monomial integral over `O(n)`, by the Gorin-Lopez recursion, in exact rationals |
| `gl2.py` | the `GL(2)` matrix coefficients, and the scalars of Section 3.1 of de Laat, Leijenhorst and de Muinck Keizer (Hermite and row-reduced normal forms over `Z` and `Q`) |
| `ps_build.py` | builds the three factors of the integrand for every `(lambda, k1, k2)` and writes them for the kernel |
| `psker.c` | the kernel: the triple walk, the memoised integral, exact rational accumulation (needs GMP) |
| `spec_split.py` | shares the entries between processes and skips ones already done |
| `zonal.py` | reduces `P(S)` modulo `S^T S = I` to the top-left block, and reads an entry of `Z_lambda` off at given inner products |
| `verify45.py` | steps 4 and 5: builds the four constraint polynomials and checks that they vanish |
| `ps_ref.py` | a slow, obviously correct version of the step 3 integral, for checking the kernel |
| `check_psd.py` | checks that `Z_lambda` is a positive semidefinite kernel on random point sets |
| `merge4.py`, `combine4.py`, `partio.py`, `run_sos4.sh` | add the pieces of the four-point constraint, which is too large to hold in one process |
| `fig_verify.py` | the figure of the two steps, drawn from the counters in `runs/` |

## Running it

```
gcc -O2 -o psker psker.c -lgmp -I/usr/include/x86_64-linux-gnu
python3 o4.py                                   # the integral self-test
python3 ps_build.py spec.bin                    # about 5 minutes, 1.5 GB on disk
./psker spec.bin ps.txt                         # about 2 hours on one core
python3 verify45.py /path/to/proofs/4_24 ps.txt # steps 4 and 5
```

`spec_split.py` splits `spec.bin` across cores; concatenating the outputs of the
parts gives the same `ps.txt`.

## What it is checked against

The construction is not checked against the authors' output.  It is checked
against four things that do not depend on it.

1. **The integral.** `o4.py` reproduces the seven values in the test distributed
   with the authors' `integrate_orthogonal.jl`, and satisfies 70 further
   identities that come from the rows and the columns of an orthogonal matrix
   being unit vectors: for any exponent matrix `M` with an empty first row,
   `sum_j int(M + 2 e_{1j}) = int(M)`.

2. **The kernel against a reference.** `ps_ref.py` builds the whole product and
   integrates it the slow way.  The two agree exactly on all 27 entries with
   `|lambda| <= 5`.

3. **The Gegenbauer polynomials.** For `lambda = (k, 0)` and two single points at
   inner product `u`, `Z_lambda` must be a multiple of the zonal spherical
   harmonic kernel of `S^3`.  It is, exactly, with ratio `8^k / (k+1)^2` for
   every `k` from 0 to 14: `8^k` from the scaling of the frames and `(k+1)^2`
   the dimension of the space of harmonics of degree `k` on `S^3`.

4. **Positive semidefiniteness.** `Z_lambda` has to be a positive semidefinite
   kernel on the subsets of size at most two of any point set; nothing in the
   computation forces this. On random four-point sets on `S^3` the least
   eigenvalue is at machine precision for every signature, including those with
   `lambda2 > 0`, which item 3 does not reach.

The check that actually settles it is step 5 itself. All four constraint
polynomials come out identically zero: the one-point constraint, the two-point
constraint in one variable, the three-point constraint in three, and the
four-point constraint in six, that last assembled out of fifty sum-of-squares
blocks and a zonal part. The two sides of it end on the same 53572 monomials,
and on every one of them the sum-of-squares coefficient is the exact negative
of the zonal coefficient, both about 15700 decimal digits long, so the
difference is zero term by term. A wrong zonal matrix would not do that.

## What the run measured

| | |
| --- | --- |
| entries of `P(S)` computed | 490 |
| monomial triples walked | 7.854e9, of 2.392e11 before the parity restriction |
| distinct exponent matrices in the largest entry | 2 505 430 |
| peak memory of `psker` | under 400 MB |
| step 3, wall clock | about two hours on two cores |
| steps 4 and 5, wall clock | about three hours, the four-point constraint in pieces |
| signatures passing the semidefiniteness check | 60 of 60 |
| Gegenbauer ratio, verified for every k from 0 to 14 | 8^k/(k+1)^2 |

## The second level, numerically (exploration)

Two scripts use the zonal matrices for something other than the check: they
evaluate the second-level kernel in floating point, as a sampled programme
needs.  Neither is part of any proof.

- `level2_numeric.py DATA PSPICKLE` builds, for configurations of 0 to 4 points
  given by their Gram matrices, the linear map from the 60 kernel blocks
  (5298 unknowns) to A_2K(Q), and validates it against the deposited
  certificate (log `runs/level2_numeric.log`, 30 s):
  - A_2K(empty) = 24 and A_2K({x}) = -1;
  - A_2K({x,y}) = -sigma_2(u) to 2e-14, against the exact coefficients;
  - 0 at subsets of the root system;
  - at most -3e-6 at random admissible triples and quadruples.
- `level2_sampled.py DATA PSPICKLE MODE KAP NQ ROUNDS [WIN]` solves the
  programme with the sums of squares replaced by constraints at sampled
  configurations (Clarabel, native interface), either for the bound or for a
  certificate that charges pairs outside windows about -1, -1/2, 0, 1/2.  At
  slack 0 with 8000 quadruples one round gives 19.42 in place of 24, fresh
  quadruples violating its constraints by up to 0.23, and needs about 10 GB
  (log `runs/level2_sampled.log`).  The certificate's own blocks have
  eigenvalues from 1e-18 to 25; a faithful solve needs the sums of squares, or
  far more samples, and extended precision.

DATA is the folder `proofs/4_24` of the certificate; PSPICKLE is
`ps.txt.reduced.pkl`, which `verify45.py` writes next to psker's `ps.txt`.

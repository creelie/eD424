# Independent verification of the D4 package (v1.3.0), 24 September 2026

This report was written against v1.3.0.  The corrections it led to, in the
package and in the manuscript, are released as v1.4.0; the section
"Addendum: v1.4.0" at the end says what v1.4.0 proves and what it leaves
open, "Addendum: v1.5.0" what v1.5.0 adds (Section 2.8 of the paper and
the Lean file D4Closure.lean), "Addendum: v1.6.0" the theorem for at
most twenty-three centres within sqrt 6 and the DOI audit, "Addendum:
v1.7.0" the explicit constant epsilon_0, and "Addendum: v1.8.0" the
stability of the kissing number and the wider neighbourhood of Theorem 2.18.
Theorem, lemma and section numbers in the body are those of v1.3.0.

Scope: the manuscript `paper/D4.tex` ("The Sphere Packing Problem in
Dimension 4 and the Twenty-Four-Cell Conjecture") and everything in this
repository that supports it.  The repository was re-run from a fresh clone.
The Lean files were built with the pinned toolchain.  Every script was run in
a scratch copy, so that nothing it writes could overwrite shipped data.  Its
output was compared with the logs shipped in `multi_cap/runs/`,
`zonal/runs/` and `lean/runs/`.  The paper was compiled and checked against
the package.

## Verdict

**The computations hold up.**  Every machine check the paper relies on was
reproduced, and none failed.  The one exception is the rational measure of
Remark 7.36, which is not in the package and so could not be checked (F4).
Reproduced:

- all seven Lean verifications, with the axiom reports the paper states;
- the exact and interval-arithmetic certificates for 23 contacts
  (Theorem 7.73) and for the single-deviation cap inequality (Section 13);
- the independent checks of the de Laat–Leijenhorst–de Muinck Keizer
  certificate for 24 contacts.  These were rebuilt from scratch here: the
  zonal matrices, all four constraint identities (each holds exactly), and
  the zero set of p_2 in Lean;
- the three independent 600-cell enumerations.

Both certificates regenerate exactly from the package: the 23-contact
certificate by re-solving its SDP (byte-identical), and p_2 from the
deposited data (byte-identical).  Every generated Lean file is byte-identical
to its committed version.  The search results the paper quotes, and the
figures, reproduce.  Details are below.

**The proof does not, as written, establish the main theorems.**  The volume
results are proved for *contact configurations*: sets of directions pairwise
at least 60 degrees apart, at distance 2.  Theorems 1.5 and 1.6 are about the
Voronoi cell of an arbitrary packing.  The only bridge between the two is
Lemma 2.7 (shell localisation) and Lemma 2.9 (radial reduction), and both are
false as stated and used.  `proof_gaps/README.md` gives concrete, checked
counterexamples to the two lemmas, not to the theorems.

- **Radial reduction (Lemma 2.9).** Pulling every neighbour closer than
  2 sqrt 2 in to distance 2 does not in general produce a packing or a
  contact configuration.  One valid 49-ball packing gives a corner cell of
  volume 6.594 < 8.
- **Shell localisation (Lemma 2.7).** Neighbours at distance 2 sqrt 2 or
  more can cut the cell.  One cuts the deletion cell from 25/3 to 8.3028.

In every example the true cell still has volume at least 8.  A numerical
search found nothing below 8 in the single-deviation setting.  The theorems
may well be true.  The missing step, though, is where non-contact neighbours
enter the 24-cell conjecture, and it is a real gap, not a typo.  It should be
resolved before the paper claims the conjecture.

**The package had a handful of defects**, fixed on this branch (section
"Changes made").  The most important: `zonal/run_sos4.sh` omitted 5 of the 50
sum-of-squares blocks of the four-point constraint, and hard-coded the
authors' paths.  The shipped result is consistent, because those blocks came
from an unlogged earlier run, but a fresh run of the script as shipped could
not have reproduced it.

## Environment

| | |
| --- | --- |
| machine | Linux 6.18 x86_64, 4 cores, 15 GB, no swap |
| Lean | 4.34.0-rc2 (commit 6a10ac8c), the pinned toolchain, from the GitHub release |
| Python | 3.11.15; numpy 2.4.6, scipy 1.17.1, sympy 1.14.0, mpmath 1.3.0, python-flint 0.9.0, cvxpy 1.9.3, clarabel 0.11.1, matplotlib 3.11.2 (3.10.9 for the figure comparison), Pillow 12.3.0 |
| C | gcc 13.3.0, GMP 6.3.0 |
| TeX | TeX Live 2023 (pdfTeX 1.40.25), latexmk 4.83 |

The Lean release server and doi.org/Crossref are blocked by this machine's
network policy.  The toolchain came from the identical GitHub release asset.
The DOIs could not be re-resolved online (see "Bibliography").

## 1. Lean (all seven verifications)

| part | result | time | axioms |
| --- | --- | --- | --- |
| `D4Stress.lean` (17 theorems) | accepted | | none |
| `D4Meet.lean` (23 theorems) | accepted | | none |
| `D4Certificate.lean` (11 theorems) | accepted | | propext, Classical.choice, Quot.sound |
| `D4InnerProducts.lean` (7 theorems) | accepted | | propext, Classical.choice, Quot.sound |
| `D4RootLattices.lean` (7 theorems) | accepted | | propext + native_decide |
| (the five files, `run_all.sh`) | | 1 min 27 s | |
| `cell600/` (16 theorems) | `lake build` succeeds | 2 min 29 s | 13 kernel theorems: none (`cells_rootlike`: propext); 3 enumeration theorems: propext, Quot.sound + native_decide |
| `certificate/` `domain_ok` | `lake build` succeeds; `verifyDomain = true` | 1479 s for `D4CertMain` | propext, Classical.choice, Quot.sound + native_decide |

* The axiom report of `run_all.sh` is identical, line for line, to the
  shipped `lean/runs/run_all_2026-09-21.log`.
* `monomialCount` evaluates to 449, as the Python expansion says.
  `D4CertStat` prints `(0, 421881)`: status 0 after 421881 boxes, the count
  `lean/README.md` states.  The whole `certificate/` build took 44 min wall
  clock on a loaded machine.
* No `sorry`, `admit`, user `axiom`, `implemented_by`, `@[extern]` or `unsafe`
  anywhere in the Lean sources.
* The statements were read against the paper.  They say what the README says
  they say.  In particular, the LDL^T positivity test, the Sturm-sequence
  argument (with q(-1), q(1/2) != 0 proved) and the 600-cell search (the
  clique-cover pruning and the exactly-once enumeration) are sound.  So is the
  branch and bound of `D4CertDomain`: its domain discards, the omega terms
  dropped on boxes below 1/3 (omega >= 0), and the second-order Taylor form
  with inherited Hessian bounds.  What `native_decide` does not certify is
  stated in `lean/README.md`: the omega tables computed in Python, and the
  Taylor and monotonicity mathematics.
* Regenerated from their sources, `D4Certificate.lean`,
  `D4InnerProducts.lean`, `cell600/D4Cell600Enum.lean` and
  `certificate/D4CertData.lean` are **byte-identical** to the committed files.
  `llm24_p2.txt`, rewritten by the fresh run of `llm24_certificate_check.py`
  from the deposited data, is also byte-identical.  So the whole chain, from
  the 4TU archive through p_2 to the kernel-checked Lean file, reproduces
  exactly.

## 2. The certificate for 24 contacts (Theorem 7.25, Proposition 7.29)

* Archive: the two parts in `third_party/llm24-certificate/` join to
  `LasserreSphericalCodes.zip`, MD5 `02acd5270f7b3fa799abdeb5291706fd`
  (matches), 331 files under `proofs/4_24`.
* `llm24_certificate_check.py` (steps 1, 2, 4, 6, 7): 10 of 10 checks pass
  in 641 s.  The numbers the paper quotes all come out: 127 blocks (60 + 2 + 15
  + 50), total dimension 3726, largest 350, least Cholesky pivot
  1.38 x 10^-15 at 256 bits, objective exactly 24, p_2 of degree 16 vanishing
  exactly at -1, -1/2, 0, 1/2.  The output matches the shipped log except for
  its last two lines, which the current script words differently; the shipped
  log was made before that wording changed.
* `p2_zeroset_check.py` (Fractions, no FLINT): zero set {-1, -1/2, 0, 1/2},
  multiplicities 1, 2, 2, 1, q < 0 on [-1, 1/2].
* `D4InnerProducts.lean`: see section 1.
* `zonal/`, step 3 (the zonal matrices), rebuilt from scratch:
  `ps_build.py` (490 entries, 1509 MB) and `psker` on two cores, 46 min wall
  clock.  The integral self-test of `o4.py` passes.  For every one of the 490
  entries, the kernel's counters (monomial triples walked, nonzero triples)
  are identical to the shipped `zonal/runs/step3_part*.log`.  That is 7.854e9
  triples in all, as the paper says.
* `zonal/`, step 5 (the four constraint identities), with `verify45.py` on
  the fresh `ps.txt`: **all four hold exactly.**
  - Constraints 1 and 2 (9 min) and constraint 3 (9 min) have 0 nonzero
    coefficients.
  - For constraint 4, the zonal half (`verify45.py ... 4 zonal`, 125 min under
    load) ends on 53572 monomials, as the paper says, with the shipped term
    counts at every checkpoint.  The sum-of-squares half, computed with the
    corrected `run_sos4.sh` over all 50 blocks in two lanes (`SOS_ONLY`),
    also leaves 53572.  `combine4.py` adds the three pieces to **0 nonzero
    coefficients: CONSTRAINT HOLDS EXACTLY**.  Unlike the shipped run, every
    block here was computed within the logged run.
* Checks of the zonal matrices themselves:
  - `ps_ref.py`: the slow reference integral agrees exactly with the kernel on
    all 27 entries with |lambda| <= 5.
  - `check_psd.py`: 60 signatures, none with a negative eigenvalue; output
    identical to the shipped `zonal_psd_check.log`.
  - `gegenbauer_check.py` (new, F12): Z_(k,0) = (8^k/(k+1)^2) U_k(u) exactly,
    for every k from 0 to 14.

## 3. The certificate for 23 contacts (Theorem 7.73)

* `certificate_check.py`: PASS.  Exact LDL^T of the nine matrices, the bound
  0.0929 > 8 - A_* = 0.092855570294 in interval arithmetic, 449 monomials, and
  the branch and bound to level 44 with every box verified.  The output is
  identical to `multi_cap/runs/certificate_check_d8.log` apart from timings.
* `D4Certificate.lean` and `certificate/` (`domain_ok`): see section 1.  The
  Lean branch and bound is a second, independent implementation.
* The certificate itself regenerates exactly.  `three_point_sdp.py 8 30
  CLARABEL 5 0.0929` reproduces every slack of the five rounds in
  `multi_cap/runs/three_point_sdp_certificate_d8.log`, and writes a
  `certificate_d8.npz` byte-identical to the committed one.  So the whole
  chain is reproducible from nothing but the code: SDP solve, certificate,
  generated Lean files (byte-identical), kernel and `native_decide` checks,
  and the interval check.

## 4. Other exact results

| script | claim | result |
| --- | --- | --- |
| `cap_certificate/cap_inequality_certificate.py` | Section 13, 21 checks | 21 of 21 PASS |
| `cell600_exact.py`, `cell600_enum.c`, Lean `cell600/` | Proposition 7.63: 0, 5, 115 | all three agree; Python and C logs identical to shipped |
| `octahedral48_exact.py` | 48 unit quaternions | identical to shipped log |
| `root_lattices_rank4.py` | census 1, 3, 23, 393; maxima 2, 6, 12, 24 | identical to shipped log; agrees with `D4RootLattices.lean` |
| `rigidity24.py` | constant stress on 96 pairs | identical to shipped log |
| `root_deletions_exact.py`, `rigidity23.py`, `rigidity_spectrum.py`, `three_point_reduction.py`, `covering_bound.py`, `extendability.py`, the `m24_*` scripts, `staged_*` | as in the README | all exit 0 with no FAIL; `three_point_reduction.py` identical to shipped log |

All 42 scripts the paper names (batch 1) ran to completion, exited 0 and
printed no FAIL, Traceback or MISMATCH.  Three of the `m24_*` scripts rewrite
floating-point SDP outputs in `continuation_out/` (`m24_primal_d6.npy`,
`m24_dir_d10/12/14.npy`).  The regenerated files differ from the committed
ones, as solver output from a different cvxpy/Clarabel version will.  The
paper treats them as exploration, and `m24_exact_reduction.py` passes on both
the committed and the regenerated primal measure.  See also finding F4.

## 5. The rest of the package (batch 2)

All 165 other Python scripts of the package (arc1_v1w1/, arc2_w1v2/,
hessian_multidir/, misc/, swap_configs/, verification/, core/, and the
remaining multi_cap/ scripts) were run with a 15-minute limit each (an hour
for the `*_derive.py` steps).  They shared the machine with the step-5 run
and were lowered in priority for part of the time, so a timeout here does not
mean a hang.

* 159 ran to completion and exited 0.  No log of any of them contains FAIL,
  Traceback, NONZERO or MISMATCH.
* All six `*_derive.py` steps completed.  The README warns they "can run for
  an hour or more".
* `swap_exact_volume.py` hit the 15-minute limit under load.  Rerun alone
  with an hour, it completes (877 s).
* The other five that hit the limit are the long searches, rerun with longer
  limits in batch 3 below.
* The cached intermediates in `data/` that the scripts regenerate reproduce:
  29 files byte for byte, including every `*_derive.py` output, and the other
  4 equal in value (`logs/scripts_other/data_cache_comparison.log`).

Batch 3, the long exploratory searches the paper names:

| script | result |
| --- | --- |
| `spherical_code_23.py` | reproduces every number of Remark 7.62 digit for digit |
| `inradius_search.py 300 7` | same result as the shipped log (94 feasible, all deletions, g <= 1/2) |
| `symmetric_search.py 16 1` | all 73 orbit structures, largest g = 0.5, none above 1/2 (Proposition 7.46) |
| `slack_continuation.py 40 1` | same picture as Table 3 (F10) |
| `multi_cap_reformulation.py` | completes (the authors' own run stopped it at 240 s) |
| `three_point_sdp.py ... 0.0929` | reproduces the certificate byte for byte (section 3) |
| `three_point_sdp.py 8 30 CLARABEL 5` | two-point rounds identical; three-point rounds fall back to SCS (F9) |
| `truncated_volume.py` | identical to the shipped `truncated_volume_r4.log`, an hour of optimisation |

## 6. Figures

All ten figure scripts in `paper/figures_new/` run and pass their label
collision tests.  With matplotlib 3.10.9, the version the shipped figures
were made with, every one of the ten PNGs is **pixel-identical** to the shipped
file.  The bytes differ only in the PNG encoding, which depends on the local
Pillow/zlib.  Under matplotlib 3.11.2 the renderings differ slightly.  The
other 16 figures of `paper/figures/` cannot be regenerated from the package:
`figures_paper/README.md` says the single panels they were composed from
are not part of it.  They are unchanged.

## 7. The manuscript

* `latexmk -pdf D4.tex` on TeX Live 2023: 147 pages, no errors, no LaTeX
  warnings, no undefined or multiply defined references or citations, no
  overfull boxes, no `??` in the output.
* The text of the rebuilt PDF is identical to the shipped `paper/D4.pdf`
  except for the date (`\today`), so the shipped PDF is current.
* All 68 files the paper names exist in the package.
* All 189 references of the form "Theorem 7.73", "Section 7.5.3", "(7.33)"
  in the READMEs, Lean sources, scripts and logs were resolved against the
  compiled paper.  All point at an item of the right type, and a scan of the
  titles found them pointing at the right result.  There was one stale number
  (fixed): `D4Stress.lean` called the equilibrium relation (7.29); it is (7.33).
* Quoted numbers.  Every decimal of six or more significant digits in the
  paper (198 of them) was looked for in the fresh run outputs, the shipped
  logs and the code (`numscan.py`).  Of those not found verbatim:
  - checked here as arithmetic or closed forms:
    (pi^2/16)^(1/4) = 0.886226, 32/23, 2/sqrt 577, 7.907144430 + 65 x
    0.001445409 = 8.001096 (and 64 pairs: 7.999651), 88 x 0.00565947,
    84 x 5.65e-5, 0.0929000002 - 0.0928555702 = 0.0000444, f_0 = 0.000927
    (from the certificate), 16/pi^2 - 1.543643 = 0.077495;
  - recomputed independently: omega(62 deg)/omega(60 deg) = 0.555035 and
    omega(66 deg)/omega(60 deg) = 0.090772 (paper 0.55504, 0.09077);
    sup_R F = 0.765832 at (3/4, 1/4, 0, 0) (paper 0.76583); and
    E[(1+q)^-2] = 0.5182543879 by deterministic quadrature (paper 0.518254,
    hence 1.543643 and 4.78 per cent).  On that last one the paper is more
    accurate than the package: `local_cell_obstruction.py` estimates it by
    Monte Carlo as 0.518307 and prints 4.79 per cent;
  - found at lower precision in the fresh logs: 0.89372, 0.002115, 0.034340
    (8.03434), 0.1677, 0.08333;
  - literature values: 0.12914461 (Li), 0.13126 (Cohn–Elkies), 60.1398863
    (Sloane's tables);
  - not reproduced by anything in the package: the Remark 7.36 measure
    (5.9826875, finding F4); the sample values 0.20945/0.20971 and
    0.1798/0.1570 of Section 7.1; and the determinant anecdote 0.30580 /
    -5.7369 of Section 20.  The last two are illustrations, not results;
  - the search results of Remark 7.62 (0.4980170 and 60.1311 deg at m = 22,
    0.5000071 at m = 24, 0.5374065 at m = 25, and 0.5000078 with 32
    near-feasible outcomes, all deletions, at m = 23) are reproduced digit for
    digit by `spherical_code_23.py` (batch 3).

## 8. Bibliography

doi.org and Crossref are blocked here, so the 50 DOIs could not be
re-resolved online.  Checked from knowledge, every DOI of the classical
references carries the right venue, volume and pages: Viazovska, CKMRV,
Musin 2008 and 2018, Bachoc–Vallentin, Cohn–Elkies, Cohn–Kumar (both),
Cohn–Zhao, CJKT, Hales (1997, 2005, 2017), Lasserre, Parrilo, de Laat–
Vallentin, DGS, Schoenberg, Blichfeldt, Korkine–Zolotareff, Fejes Tóth,
Schütte–van der Waerden (both), Gorin–López, Voronoi, Curry–Schoenberg,
Farouki, Roth–Whiteley, Quickhull, NumPy, SciPy, SymPy and Arb.  The 2026
Clarabel entry could not be checked.

## Findings

F1 (proof; corrected in v1.4.0, see the addendum): the radial reduction.  See
`proof_gaps/README.md`.

F2 (proof; corrected in v1.4.0, see the addendum): shell localisation.  See
`proof_gaps/README.md`.

F3 (package, fixed): `zonal/run_sos4.sh` omitted blocks 0, 1, 3, 4 and 5 of
the 50 sum-of-squares blocks of the four-point constraint.  Its own comment
counts seven large blocks and gives four of them groups.  It also hard-coded
`/home/claude/aud` and `/tmp/claude-0/`.  The shipped log shows the running
total already holding 74606 terms before the first listed group, so those
blocks came from an earlier, unlogged run.  The script now takes the data
folder, `ps.txt` and a work directory as arguments, groups all 50 blocks, and
refuses to run if the groups do not cover 0..49 exactly once.  (A first
version of that guard used the variable name `GROUPS`, which bash reserves;
it is now `SOS_GROUPS`.)

F4 (paper; stated in v1.4.0): Remark 7.36 says a rational measure on 632 of 670 grid
triples, with value 5.9826875 and an exact LDL^T, *proves* that no degree-6
certificate exists.  Neither the measure nor any script that builds or checks
it is in the package.  The shipped `m24_primal_d6.npy` is a different,
floating-point measure: 2170 triples, value 5.942410 as read by
`m24_exact_reduction.py`.  A fresh run of `m24_primal_sdp.py` gives 522
triples and 5.985098.  The Remark is not on
the path of the main theorems, but as it stands its "proof" cannot be
checked from the package.

F5 (documentation, fixed): `third_party/llm24-certificate/README.md` listed
three archive parts (`-00, -01, -02`); there are two.  It also gave the size
as "234 MB" beside the archive name.  The zip is 152065368 bytes.  The 234 is
the unpacked data as `du -h` reports it (243787660 bytes).  Both lines are now
exact.  The paper's "234 megabytes in 331 files under proofs/4_24" is that
unpacked figure, and is left as it is.

F6 (documentation, fixed): the README's directory entry and usage section for
`lean/` predated `D4RootLattices.lean` ("the first four", four `lean`
commands).  `lean/README.md` said "Thirty-seven theorems across the two
files"; `D4Stress.lean` and `D4Meet.lean` have 17 + 23 = 40.  The README said
`three_point_reduction.py` takes "about twenty minutes"; it takes seconds (its
Monte Carlo is seeded), as the authors' own `named_scripts_run.log` records.

F7 (metadata, fixed): `CITATION.cff` lacked P. Mandal's affiliation, which the
paper and `.zenodo.json` give.

F8 (documentation, noted): `RELEASE_NOTES_v1.3.0.md` says every figure script
"reproduces the shipped PNG byte for byte".  That holds pixel for pixel with
matplotlib 3.10.9, not byte for byte in general.  The same notes say D4.pdf
"is delivered separately (it is 21 MB)"; it is in `paper/` and is 18 MB.  The
release notes are a record of v1.3.0 and were not edited.

F9 (reproducibility, noted): in its search mode (`three_point_sdp.py 8 30
CLARABEL 5`), the three-point solves fail under Clarabel with cvxpy 1.9.3, with
both clarabel 0.11.1 and 0.10.0.  The script then falls back to SCS, which is
slower and gives lower sampled bounds than the shipped log: 0.0946 against
0.0957 in round 1.  The two-point rounds reproduce the shipped log exactly.
The shipped log was made by an earlier version of the script (its warning
points at line 154; the SCS fallback is now at line 195), so the solver
behaviour it records cannot be recreated here.  This is exploration: the
certificate mode, which produces the certificate the proof uses, runs under
Clarabel and reproduces its log and the certificate byte for byte.

F10 (logs, noted): several shipped logs predate small script changes, so a
fresh run differs from them in form, not substance.  `llm24_certificate_check.log`
has a reworded final message; `step5_constraints_1_2.log` has older progress
lines; `slack_continuation_seed1_fresh40.log` lacks the delta levels 0.009 and
0.008 that the script (and Table 3) now include; and
`symmetric_search_16_seed1.log` stops before the final summary, which the
fresh run prints (largest g 0.5, none above 1/2).  The paper's slack table
takes the best over three passes; the fresh seed-1 pass shows the same
picture: far from deletions down to delta = 0.009, gone at 0.008, and every
endpoint a deletion at delta = 0.

F11 (code, noted): `gcc -Wall -Wextra` on `zonal/psker.c` gives one
`-Wmaybe-uninitialized` warning in `canon()`.  It is a false positive: every
byte of `t` is written before the `memcpy`.  Sorting rows and columns there
is not a unique canonical form under simultaneous permutation.  That costs
only memo hits, since the O(4) monomial integral is invariant under row and
column permutations and under transposition.

F12 (package, fixed): `zonal/README.md` (item 3) and Section 7.5.3 of the
paper say that Z_(k,0) at two single points was checked to be the zonal
harmonic kernel of S^3, "exactly, with ratio 8^k/(k+1)^2 for every k from 0
to 14".  No script in the package did this.
`independent_verification/gegenbauer_check.py` now does it, through
`zonal.py` exactly as `verify45.py` evaluates Z, and the claim holds.

F13 (code, noted): `multi_cap/local_cell_obstruction.py` estimates
E[(1+q)^-2] at the regular simplex by Monte Carlo (0.518307), prints "4.79
per cent", and labels its check "4.78 per cent".  The paper's 0.518254,
1.543643 and 4.78 per cent are the accurate values (0.5182543879 by
quadrature).  The script's tolerance still passes.

F14 (package, noted): `third_party/llm24-certificate/verify.sh` runs
`psker` in one process and all of step 5 in a single `verify45.py` process.
The zonal README says the four-point constraint was too large for one
process on the authors' machine.  This verification ran the same programs
split up, as described in section 2, and did not run `verify.sh` end to
end.

## Changes made

| file | change |
| --- | --- |
| `zonal/run_sos4.sh` | F3: all 50 blocks; coverage guard; takes the data folder, `ps.txt` and a work directory as arguments (resolved before it changes directory); `SOS_ONLY` to share the groups between processes; a failed group now makes the script exit 1 (added after the run above, which only exercised the success path) |
| `lean/D4Stress.lean` | comment: equation (7.29) -> (7.33); rechecked with Lean |
| `lean/README.md` | "Thirty-seven" -> "Forty" |
| `README.md` | `D4RootLattices.lean` in the directory entry and usage; runtime of `three_point_reduction.py` |
| `third_party/llm24-certificate/README.md` | part list and sizes (F5) |
| `CITATION.cff` | Mandal's affiliation (F7) |
| `independent_verification/` | this report, the proof-gap examples, `gegenbauer_check.py` (F12), `compare_figures.py`, `numscan.py`, and the logs of every run |

The manuscript `paper/D4.tex` was not changed by the re-verification itself.
v1.4.0 then revised it for F1, F2 and F4; see the addendum.

## Addendum: v1.4.0

The corrections of v1.4.0, which answer F1, F2 and F4, were made after this
report and are checked here in the same way.  Numbers below are those of
v1.4.0.

**What is proved, and what is not.**  Nothing in the re-verification closes
the twenty-four-cell conjecture for all packings, and v1.4.0 does not claim
it.  The bound vol >= 8, with equality only at the root system, is proved:

- for every *contact configuration*.  This is the contact-count argument,
  unchanged: at most 22 contacts (Theorem 1.3), exactly 23 (Theorem 7.40)
  and exactly 24 (Theorem 7.25);
- for the *Voronoi cell of a packing* in two cases (Theorem 1.5).  The first
  is when every centre within 2 sqrt 2 of the given one touches it.  The
  second is when the distances d_i of those centres satisfy
  Phi(d_1, d_2, ...) > 8, where Phi is the distance criterion of
  Proposition 2.10.

The case left open is Conjecture 1.6: at least 23 centres within 2 sqrt 2,
at least one of them not touching, and Phi <= 8.  The density bound pi^2/16
is proved for periodic packings whose centres satisfy the hypothesis of
Theorem 1.5 (Theorem 1.7).  For all packings it depends on the conjecture.

**The new arguments (Section 2.7).**

- *Proposition 2.10.*  The radial form of the cell volume, truncated at
  radius sqrt 2, sees only the centres within 2 sqrt 2.  Each of those
  centres removes at most a cap of radius arccos((d_i/2) cos r) at level
  sec r.  That gives vol(V_c) >= vol(V_c cap B(sqrt 2)) >= Phi(d_1, ...),
  with Phi nondecreasing in every d_i.  For contacts only, Phi is the
  covering bound of Theorem 7.16 cut off at 45 degrees.  That cut-off changes
  the bound only for m <= 11, and the value stays above 11.5 there.
- *Lemma 2.11.*  The root directions have covering radius exactly 45 degrees:
  (a + b)^2 >= a^2 + b^2 + c^2 + d^2 because 2ab >= c^2 + d^2.  So a centre
  with 24 contacts has no other centre within 2 sqrt 2, and its cell is the
  24-cell.
- *Proof of Theorem 1.5 for shell-free centres.*  For m <= 22,
  Phi > 8.044.  For m = 24, Lemma 2.11 applies.  For m = 23, the cell is
  bounded (Theorem 7.25), and Theorem 7.73 bounds its part inside
  B(sqrt(3/2)) above 8.  Centres beyond 2 sqrt 2 do not reach that ball.

**Checks.**

- `multi_cap/shell_reduction.py`, in ball arithmetic (python-flint, 160
  bits), passes every check.  Its log is `multi_cap/runs/shell_reduction.log`.
  - Phi for m contacts has rigorous lower bounds 11.5286 (m = 11) and
    8.04415 (m = 22).
  - It computes the thresholds of Table 1, for example 22 contacts plus one
    further centre at d >= 2.16756.
  - With 23 contacts and one further centre, no d below 2 sqrt 2 passes the
    criterion.
  - Both counterexamples of `proof_gaps/` are consistent with the corrected
    lemmas, and the 49-ball example satisfies Phi >= 18.3358.
  - Each value of Phi is a lower Riemann sum.  The decreasing factor is taken
    at the right end of each interval and the increasing factor at the left,
    so every printed value is a lower bound.
- The exact gradient of the polytope volume used by
  `multi_cap/shell_neighbour_search.py` agrees with finite differences to
  1e-8.  It reproduces the volumes 8 (the root system) and 25/3 (a deletion).
- The search of the open case, which is exploration and not proof:
  - Free search, 240 starts (`shell_neighbour_search_free_240_seed1.log`):
    239 endpoints were feasible.  224 of them brought every non-contact into
    contact or beyond 2 sqrt 2, and the least volume among them was
    8.000000000, at the root system.  The other 15 kept a centre in the
    shell, and the least volume among those was 8.336218986 (22 contacts,
    one centre at 2.3986).
  - One neighbour held at distance 2 + delta, 60 starts for each of eight
    values of delta (`shell_neighbour_search_delta_*_60_seed7.log`):

    | delta | feasible | least volume | contacts | on-axis value 25/3 - (1/3)(1 - delta/2)^4 |
    | --- | --- | --- | --- | --- |
    | 0.001 | 59 | 8.000666167 | 23 | 8.000666167 |
    | 0.005 | 59 | 8.003320854 | 23 | 8.003320854 |
    | 0.02 | 56 | 8.013134663 | 23 | 8.013134663 |
    | 0.05 | 53 | 8.032104036 | 23 | 8.032104036 |
    | 0.1 | 54 | 8.061831250 | 23 | 8.061831250 |
    | 0.2 | 55 | 8.114633333 | 23 | 8.114633333 |
    | 0.4 | 55 | 8.196800000 | 23 | 8.196800000 |
    | 0.8 | 52 | 8.283732535 | 22 | 8.290133333 |

    For every delta up to 0.4, the least volume found is the on-axis value:
    the deletion of a root, with the held neighbour on the deleted root's
    axis.  No endpoint fell below 8.  As delta tends to 0 the values tend to
    8, so the open case is tight near the 24-cell.

**The manuscript.**  The revised paper was built with `latexmk -pdf`: 0
undefined references, 0 overfull boxes, and no "??" in the text.  Every new
cross-reference resolves, and every number quoted in Section 2.7 matches
`multi_cap/runs/shell_reduction.log`.  The title-page note, abstract,
introduction, Theorems 1.1, 1.5 and 1.7, Section 6, the conclusion, and the
data and code availability statement are revised to the claims above.
Theorem 1.1 is now stated for the cell cut out by the active neighbours,
under the hypothesis its proof uses: the deviating direction replaces a root
that no active neighbour takes.

## Addendum: v1.5.0

v1.5.0 narrows Conjecture 1.6 in a new Section 2.8 of the paper, and adds a
Lean layer for its exact content.  It does not prove the conjecture, and the
paper does not claim that it does.  The case still open is a centre with at
least 23 other centres within sqrt 6, one of them at a distance between
2 + epsilon_0 and sqrt 6, where epsilon_0 is not explicit.

Checks:

- **`multi_cap/closure_lemmas.py`** passes every check; the log is
  `multi_cap/runs/closure_lemmas.log`.  It checks symbolically the
  identities of Lemma 2.12, Proposition 2.13 and Lemma 2.15.  It checks in
  integers the root-pair combinatorics of Corollary 2.14:
  - 72 orthogonal pairs, three at each of the 24 vertices;
  - no deleted set of one or two roots meets all three pairs at any vertex;
  - of the 2024 deleted triples, 96 do, and so do 1728 of the 10626 deleted
    quadruples.

  It computes in ball arithmetic S(2) = 0.138963500936 and
  9 pi^2/8 - 22 S(2) = 8.04610793064 > 8.046, the thresholds of Psi (within
  0.003 of those of Phi), and arccos(sqrt 6 / 4) = 52.238756 degrees.
- **`lean/D4Closure.lean`** proves the same exact content again: the four
  identities over every commutative ring (`grind`) and the combinatorics by
  `decide +kernel`.  It has no `sorry` and no Mathlib.  The axioms are
  propext, Classical.choice and Quot.sound at most, and nothing uses
  `native_decide`.  `lean/run_all.sh` checks all six standalone files in
  3 minutes; the log is `lean/runs/run_all_2026-09-25.log`.
- **`multi_cap/near_contact_probe.py`** is exploration, not proof; the log
  is `multi_cap/runs/near_contact_probe_200_seed5.log`.  On the root system
  pushed out by delta, the inversion-hull bound and the volume agree to
  first order, (2/3) sum delta.  Over 200 volume minimisations near the root
  system with one centre held out:
  - ||eps|| / kappa is at most 0.000262, where the proof of Theorem 2.18
    allows 2.71;
  - (vol - 8 - (2/3) sum delta) / (||eps||^2 + |delta|^2) is at least -0.5,
    the value on the deleted root's axis.
- **The proof of Theorem 2.18** was read against these numbers.  Its three
  estimates are the rigidity with slack, the first-order volume expansion
  and the hull's fourth-order cost.  Its constants come from the explicit
  spectrum of Lemma 7.50, from a Lipschitz stability argument for polytopes
  with an interior point, and from compactness; only the last makes
  epsilon_0 non-explicit.
- **The manuscript** builds with `latexmk -pdf` to 157 pages: no undefined
  references, no overfull or underfull boxes, and no "??".
- **F4 (Remark 7.36)** is resolved in the text.  The remark no longer
  claims a proof from a rational measure that the supplement does not
  carry; it states the floating-point evidence and says that nothing
  depends on it.
- **The bibliography:** `check_bibliography.py` finds all 57 entries
  cited, no undefined citation, no duplicate key, and every one of the 51
  DOI links equal to the DOI printed beside it.  Live resolution against
  Crossref and DataCite (`--online`) was not possible here: this
  environment's network policy denies doi.org and api.crossref.org.  The
  DOIs are unchanged since v1.3.0, whose release notes record that every
  one was resolved and compared with its metadata.  The exceptions are the
  two Zenodo DOIs of this package.

## Addendum: v1.6.0

v1.6.0 proves Theorem 2.17: a centre with at most 23 other centres within
sqrt 6 has vol(V_c) > 8, whatever their distances.  It replaces Corollary
2.17 of v1.5.0 (twenty-three contacts), which it contains.  Conjecture 1.6
is narrowed to a centre with at least 24 other centres within sqrt 6, one of
them at a distance between 2 + epsilon_0 and sqrt 6.  It is not proved, and
the paper does not claim that it is.

Checks:

- **`multi_cap/labelled_certificate_check.py`** passes every step; the log
  is `multi_cap/runs/labelled_certificate_check.log` (about six minutes).
  - It rebuilds the certificate of Theorem 7.73 and B = 0.0929000002
    exactly, and rechecks the positivity of the matrices.
  - In ball arithmetic it gets s(D) = 0.0928820620 > 8 - A_* for
    D = 2.1648, a_D = 0.5732294553, a_D / (1 - (D/2) a_D) = 1.5103 < 2,
    fr(1, 1, 1/2) = 0.02371033 < 1/22, r >= 0.03790036,
    kappa >= 0.69636299 and c >= 0.105569628 (in units of 1/1000,
    105.569628).
  - It reruns the branch and bound of (C) (313 780 boxes).
  - It runs the new ones on 1/2 <= t <= 0.51 (30 051 boxes verified,
    3 408 outside the domain) and on 0.51 <= t <= a_D (6 482 verified,
    1 242 outside).
- **The reduction** Q >= Q_0 + Gamma_1 + Gamma_2 + Gamma_3 and the table
  bound were tested against the true labelled quantity, the six-variable
  function computed by quadrature.  The test used 1 600 random admissible
  triples with t > 1/2 and 1 500 in the slab.  There was no violation: the
  true value was never below either bound, and the least gap in the slab
  was 0.00019.  At the corner d_i = D, u = v = t = a_D the true value is
  11.02 (in units of 1/1000), against -1.85 for Q_0 alone.
- **`multi_cap/truncated_search.py`** is exploration, not proof.  Its
  closed form of the pair terms agrees with quadrature to about 1e-15.  It
  minimises the right side of Lemma 2.15 with exactly M centres within
  sqrt 6, from twelve starts each, of which four or five end at a local
  minimum; the logs are `multi_cap/runs/truncated_search_M24.log` to
  `_M27.log`.
  - M = 24: every local minimum found is the root system, 7.906940.
  - M = 25 to 27: every local minimum found lies above 8.26.
- **`lean/D4Closure.lean`** proves two more theorems:
  - `amax_tangent` (grind, three standard axioms);
  - `triple_counts` (decide, no axioms).

  `lean/run_all.sh` passes; the log is
  `lean/runs/run_all_2026-09-25_v1.6.0.log`.
- **The figures.**
  - `fig_labelled.py` passes the label collision test (0 collisions).
  - `tikz_labelled.tex` and `tikz_stop.tex` were rendered at 500 dpi and
    inspected: no label touches a line or another label.
- **The manuscript** builds to 162 pages, with no undefined references and
  no overfull or underfull boxes.  No sentence of more than 90 characters
  occurs twice.  The two repeated attributions of the classification of
  24-point codes, in the introduction and the conclusion, are reworded.
- **The bibliography** was checked entry by entry against the publishers'
  and indexers' pages found by web search.  doi.org, api.crossref.org,
  api.datacite.org, arxiv.org and data.4tu.nl are not reachable from here.
  The record is `logs/doi_audit_2026-09-25.md`.
  - 49 of the 51 DOIs are confirmed with matching title, authors, volume,
    year and pages.
  - BBB26 is confirmed apart from its version suffix `.v11`, which the
    authors should check against the current version on Preprints.org.
  - LLM24data (4TU.ResearchData) is consistent with the archive downloaded
    through it, which matches the recorded MD5, but is not indexed by the
    search engines.

  No DOI points to a different work.

## Addendum: v1.7.0

v1.7.0 makes the constant of Theorem 2.18 explicit: epsilon_0 = 4e-26.
Before, it came from a compactness argument.  Conjecture 1.6 is still
open, and the paper does not claim otherwise; what is left keeps its shape,
a centre with at least 24 other centres within sqrt 6, one of them at a
distance between 2 + epsilon_0 and sqrt 6.

What is new, and how it was checked:

- **Theorem 7.76 (twenty-four points, approximately).**  The sum-of-squares
  identities of the certificate of de Laat, Leijenhorst and de Muinck
  Keizer hold as polynomial identities, and the certificate kernel is a
  Gram kernel on every subset of at most two points, so the chain of
  Lemma 7.27 holds for any finite set of directions.  If the inner products
  may exceed 1/2 by kappa, only the terms whose weight is an elementary
  symmetric function of the pair factors (u + 1)(1/2 - u) can be negative.
  `multi_cap/explicit_eps0.py` classifies every prefactor of the deposited
  data (constants, Gram determinants, principal minors, or those
  functions), and for the sign-changing ones bounds, in ball arithmetic,
  the largest eigenvalue of the block (trace or row sum) and the Chebyshev
  l1 norms of the transformed vectors.  Result: B_3 <= 8.75e7,
  B_4 <= 7.34e9, and E(24, 2e-26) <= 2.35e-12.
- **The two-point polynomial.**  sigma_2 is divided exactly by
  (u+1)(u+1/2)^2 u^2 (u-1/2); the quotient is bounded below by 2.62e-4 on
  [-1, 1/2] with a mean value form on 30 000 intervals (a floating-point
  scan on 200 001 points gives the minimum 2.6209e-4 at u = -0.8575).
- **Lemma 7.77.**  The minor bound (rho + 2 delta sqrt 5)^5 - rho^5 = 0.81
  < 1 at delta = 3e-4 is checked in ball arithmetic.  The distance bound
  was also tested in floating point on 2 000 random perturbations of the
  normalised roots with Gram error below 3e-4: the Procrustes distance was
  at most 0.073 times the bound.
- **Theorem 2.18.**  The facet and hull estimates are evaluated in exact
  rational arithmetic, at an upper bound of Theta; every expression is
  increasing in Theta.  The bracket is positive at S <= 24 epsilon_0 and
  at S = 4e-5 (0.0204).

The run takes sixteen seconds; the log is `multi_cap/runs/explicit_eps0.log`.

`multi_cap/truncated_search.py rays` (floating point, exploration) locates
where the pair terms of Lemma 2.15 reach 8 near the root system: at
sum delta_i = 0.1551 with all 24 centres pushed out evenly, and at
delta = 0.1971 with one centre pushed out.  The explicit neighbourhood of
Theorem 2.18 ends at sum delta_i = 4e-5.  The log is
`multi_cap/runs/truncated_search_rays.log`.
The paper builds to 166 pages with no undefined references and no overfull or
underfull boxes; the new section is 7.13, at the end of Section 7, so no
theorem, lemma or section number changes.  The new Figure 25
(`paper/figures_new/tikz_eps0.tex`) was rendered at 500 dpi and inspected: no
label touches a line or another label.

## Addendum: v1.8.0

v1.8.0 adds a theorem, sharpens two constants, and corrects one printed
value.  Conjecture 1.6 is still open, and the paper does not claim
otherwise.  The results came to this package as a separate submission (a
certificate package and a technical note); every part was rerun or
recomputed here before it went into the paper, and three things in it were
corrected.

What is new, and how it was checked:

- **Theorem 7.79 (the kissing number is stable).**
  - `multi_cap/certify_cardinality.py` passes on both degree-8
    certificates, at s = 0.0065 and at s = 0.005, in about 80 seconds each.
    It checks exact LDL^T positivity of every F_k and computes
    f(1) + F(1,1,1) exactly (23.974999950 at s = 0.0065).  Its interval
    branch and bounds reuse `certificate_check.py`: 150 225 boxes at
    s = 0.0065, and the univariate check.
  - The derivation of the bound from (i) and (ii) was redone by hand: the
    sum over ordered pairs and triples, then a 25-point subset of any
    larger set.  The subset replaces the monotonicity argument of the
    submission, which fails for very large |C|.
  - The degree-10 certificate at s = 0.008, the one the theorem states,
    passes as well: f(1) + F(1,1,1) = 23.959999949, e2 = 7e-5, 2 642 578
    boxes in 48 minutes, and 24(1 - e1) - 552 e2 = 23.961336 > 23.96.  The
    shipped log is a second run with the final version of the checker.
- **Correction 1: the threshold.**  The submitted checker ran its boxes to
  the double nearest 0.5065, which lies 5e-17 below 0.5065.  The checker
  now takes the recorded decimal exactly and runs to the least double at
  or above it.  It also rounds e1 and e2 down to doubles, so that
  "P <= e" holds as printed.  The rerun passes.
- **Lemma 7.50, second statement.**  `multi_cap/rigidity_spectrum.py`
  checks it in integers:
  - 2K = Lambda'(2P)Lambda'^T annihilates x(x-4)(x-10)(x-12)(x-16).
  - (2K-4I)(2K-10I)(2K-12I)(2K-16I) has every diagonal entry 2400, which
    is 7680 times 5/16.
  - The trace of the projection onto the image is 66.

  The symmetry argument of the text gives the same value.
- **Correction 2: the l1 constant nu.**  The technical note reported
  nu = 3.361 from 400 Nelder-Mead restarts.  That is not the minimum.
  - A single displaced direction, with its rotational part removed, has
    ratio sqrt(96/11) = 2.954, computed in rational arithmetic.
  - Three hundred linear programmes of the form "minimise ||Lambda x||_1
    subject to <c, x> = 1" find nothing lower.

  The paper states only the proved facts: nu >= 4/sqrt 11 and
  nu <= sqrt(96/11), so this argument cannot give a radius above 0.244.
- **Correction 3: 2/sqrt(397).**  The submitted patch printed 0.1003774;
  the value is 0.1003771.
- **Theorem 2.18.**
  - Estimate (a) was sharpened and rederived by hand.  The positive parts
    of g = Lambda tau are at most a entrywise, and the negative parts sum to
    at most A = sum a.  With ||g||_inf <= (sqrt 11/4) ||g||_2 this gives
    ||g||_2 <= (sqrt 11/8) A + ((11/64) A^2 + ||a||_2^2)^(1/2), hence
    ||eps|| <= 2.129 (1 + delta) S + 9.49 ||eps||^2.  `multi_cap/explicit_eps0.py`
    uses rational upper bounds for sqrt 11, sqrt(27/16) and sqrt(75/64), and
    concludes ||eps|| <= 2.3674 (1 + delta) S at S = 4e-3.
  - Step (b) of the proof is new and was checked in three ways.
    1. The derivation was redone by hand.  The facet formula of step (b) is an
       identity, and the two facet bounds follow from face-by-face
       comparisons with the octahedron.
    2. `multi_cap/facet_bounds_probe.py` confirms the identity numerically
       (relative 1e-7) and the facet bounds on 4320 facets of random and
       adversarial perturbations.  The bounds are used to at most 82 and
       14 per cent.
    3. `explicit_eps0.py` evaluates the resulting bracket in exact
       arithmetic.  It checks that the coefficient of |eps|^2 in the losses
       exceeds 2/3, so that the largest |eps| is the worst case.
  - The bracket is 0.1355 at S = 4e-3, and the theorem now states that
    neighbourhood.
- **Remark 7.78.**  `multi_cap/robust_ceiling.py` recomputes L2 = 0.22572
  and q_min = 2.6209e-4 from the exact sigma_2 and gives the ceilings
  4.73e-14 (delta = 3e-4) and 9.24e-9 (delta = 1/4).
- **The erratum.**  7.906940 at the root system truncated at r_*: confirmed
  against `multi_cap/runs/truncated_search_M24.log` and by
  `rebuilt_from_text/d4_independent_check.py`.
- **The 25-point code.**  `multi_cap/code25_search.py` finds 25 points with
  largest inner product 0.537429, written to
  `multi_cap/runs/code25_best.txt` and rechecked from the file.

`rebuilt_from_text/d4_independent_check.py` recomputes, from the statements
of the paper alone:

- the covering bounds of Table 2;
- the four pair-bound values;
- the rigidity spectrum and the 11/16 diagonal;
- the crossings 0.15511 and 0.197054;
- the crossing 0.147332 with the truncation radius grown with the
  distances.

It agrees with the paper on every item.  Its log is beside it.

The paper builds with no undefined references and no overfull boxes.

- **Numbering.**  The new results are Remark 7.78 and Theorem 7.79, at the
  end of Section 7.13, and the 11/16 statement is part of Lemma 7.50.  No
  theorem, lemma or proposition number of v1.7.0 changes; the new
  Figure 26 renumbers the later figures by one.
- **Figures.**  Figures 16 and 25 were redrawn and Figure 26 is new.  The
  two matplotlib figures pass the label-collision test, the TikZ figure was
  rendered at 500 dpi, and all three were inspected.

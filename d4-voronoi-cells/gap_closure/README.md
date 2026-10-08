# gap_closure: floating-point explorations of the open cases

This directory holds the computations run against the two statements that the
paper leaves open (Section "The statements (G) and (C)"): statement (C) for 25
to 28 centres within sqrt6 and statement (G), and the three cases they
settled, the residual case at thirty centres, the whole case at twenty-nine,
and twenty-eight with at most fourteen centres within 2.0161 (fifteen with
at most three beyond 2.35, sixteen with none). They are
floating-point semidefinite programmes on sampled constraints. **Three of them
yield certificates**: the round-3 certificate of `C30/combo30p.py`, which
`multi_cap/combo30_check.py` proves exactly and which is the fifth case of
`thm:count30`, and the certificates of the runs `c29r4` and `c28lo13b` of
`CM/combo_direct.py`, which `multi_cap/combo_case_check.py` proves exactly and
which are `thm:count29` and `prop:count28-few`. The rest are kept so that the
values quoted in the paper and in the pull requests can be reproduced, and no
result of theirs is used in any proof, except the exact checks logged in
`density/check_M.log` (see the last section).

The level that (C) asks for is `9 pi^2/8 - 8 = 3.10330`: a programme proves a
case only if its bound, after the correction for the constraints that the
samples miss, is below that level.

## C30: the residual case at thirty centres (proved)

The fifth case of `thm:count30`: 24 centres within 2.25 (22 within 2.05, 23
within 2.15) and 6 beyond 2.4. The two-point programmes of the tree do not
close it (`rem:thirty-pairs`); the kernel on typed triples does.

| script | kernel | result (log) |
|---|---|---|
| `combo30.py` | distance-labelled two-point kernel plus a three-point kernel typed by distance class (A = [2, 2.05], B = (2.05, 2.25], F = [2.4, sqrt6)) | degree 0: 3.1212 (`combo30_d0.log`); degree 2: 3.1211 (`combo30_d2.log`); degree 6, three rounds: raw 3.0991, corrected 3.1160 (`combo30_d6b.log`) |
| `combo30p.py` | the same, with the sample set pruned so that more rounds fit in memory | degree 6: raw 3.0995, corrected 3.1103 (`combo30p_d6.log`); degree 8, two rounds: raw 2.8946 then 2.9338, corrected 4.4901 then 3.8157 (`combo30p_d8.log`, `combo30p_d8b.log`); resumed with 3000 pair and 1200 triple samples kept per kind (`combo30q_d8.log`): raw 2.9349, 2.9388, 2.9414, 2.9422 and corrected 3.3005, 3.1528, 3.0062, 2.9971 in four rounds; the round-3 certificate `combo30q_d8_r3.npz` is the one proved (copied to `multi_cap/radial_certificates/combo30_d8.npz`) |
| `typed_delsarte.py`, `code_feasible.py`, `code_feasible2.py`, `code_feas_gen.py` | typed linear programming bound and local search for two-shell codes with points in holes | exploration only (`slackhole_d6.log`) |
| `case_value.py` | value of the two-point case programme of the paper for given count constraints | used to choose splits |
| `multi_cap/combo30_check.py` | the exact check that a certificate of `combo30p.py` has to pass: exact positivity, the pair inequalities by Bernstein branch and bound with Pi in Arb, the bins, the typed triple inequalities by Taylor branch and bound, and the bound over the five count vectors | **PASS** on `combo30q_d8_r3.npz` with margins 2e-5, 5e-6, 2e-6: 47 646 pair boxes, 4131 bin intervals, 8 275 056 triple boxes, largest bound 3.087203 at counts (22, 1, 1, 0, 6) < 3.10330 (`C30/combo30_check_r3.log`, 6891 s) |

The raw value is the optimum on the samples; the corrected value adds, for each
kind of pair and triple, the largest violation found between the samples times
the number of such pairs or triples. At degree 6 the raw value rises with each
round of added samples and the corrected value stays above 3.10330.

## CM: twenty-five to twenty-nine centres (twenty-nine proved)

`combo_gen2.py` generalises `combo30.py` to an arbitrary case file
(`case29_all.json`: 29 centres, types A, B, F, seven bins and the count
constraints proved by `prop:C-radial`), and bounds the maximum over all count
vectors by linear-programming duality instead of enumerating them
(`combo_gen.py` enumerates them and runs out of memory). With the two-point
kernel alone (degree 0 in the three-point part) the value is 3.1967
(`c29all2_d0.log`), far above the level.

`combo_direct.py` runs the same programme through Clarabel's own interface
instead of cvxpy, in about half the memory (environment: `DSM` the
factorisation, `THREADS`, `TAG`). The case files `case25_all.json` to
`case28_all.json` carry the constraints of `prop:C-radial` at 25 to 28
centres (5330, 17 845, 42 927 and 74 550 count vectors), and `case28_t4.json`
splits the close centres of 28 into four types. At 29 and degree 8
(`c29all_d8.log`, continued from its samples in `c29x_d8.log`) the raw value
is 3.0287, 3.0568, 3.0592, 3.0605 in four rounds and the corrected value
comes down to 3.1026, but the corrected values of these early rounds were too
low (see the next paragraph): the precheck of the round-2 certificate of the
run with the compass search (`c29r_d8.log`) gives 3.196 (`pre_c29r_r2.log`).

The run that closes 29 continues that one. Resumed with `REFINE=100`
(`c29r3_d8.log`), its raw value is 3.06337 to 3.06413 in four rounds, with
corrected values 4.86, 3.09964, 5.78 and 3.227: pruning to the samples nearest
to violation dropped whole regions of pairs, which came back as large
violations a round later. With `KEEP_BASE=1`, which keeps the starting grid of
pair samples through every pruning, the resumed run `c29r4` (`c29r4_d8.log`)
gives raw 3.06421 and corrected 3.10016 in its first round. The precheck of
that certificate at margins 5e-6, 5e-7, 1e-6 is 3.099635
(`pre_c29r4_r1_small.log`), and the exact check
`multi_cap/combo_case_check.py case29_all.json
radial_certificates/combo29_d8.npz 8 5e-6 1e-6 1e-6` passes with 3.101462 <
3.10330 (`multi_cap/runs/combo29_check.log`, 104 minutes): this is
`thm:count29`, and `combo29_d8.npz` is `c29r4_d8_r1.npz`. Four types at 29
(`c29t4c_d8.log`) and degree 10 (`c29d10_d10.log` here, which diverged, and
`bigmachine/records/c29all_d10.log` on a 16-core machine, raw 2.9353,
3.0004, 3.0105, 3.0119 in four rounds) were not needed.

Below 29 the programme stays above the level on its samples, and samples only
lower the optimum: at 28, 3.10668 with three types (`c28x_d8.log`, one round)
and 3.11169 with four (`c28t4_d8.log`, round 2); at 27, 3.1570 (round 1 on the
larger machine, `bigmachine/records/c27all_d8.log`); at 26, 3.30766 at
degree 6 (`c26s_d6.log`) and 3.2456 at degree 8 (`c26h_d8.log`). At 29 degree
10 lowers the raw value by about 0.05 against degree 8, so 28 is run at
degree 10 next. Runs on a larger machine go through
`bigmachine/run_combo.py`.

Each round of `combo_gen2.py` (and so of `combo_direct.py`) looks for the
violations between the samples by random points and grids and, unless
`REFINE=0`, by a compass search from the worst of them and from the worst
samples (`REFINE` starts per kind, default 60), which only moves to admissible
points of larger value. On the round-2 certificate at 29 that search finds
violations two to four times larger than the random points for the triples
(for example 1.8e-4 against 8.0e-5 for ABF), which is why the corrected values
of the earlier rounds were too low. `multi_cap/combo_case_check.py` with
`PRECHECK=1` computes the thresholds of the exact check and the bound they
give, without the branch and bounds, to say whether a certificate is worth
the full check.
Each round adds the `NEWP` worst new pair samples (default 3000) and the
`NEWT` worst new triple samples (default 400) per kind, the triples drawn from
`TPROBE` random points (default 8000) and a grid, before the compass search.
With `BORDER=1` the exact check rounds the bordered matrix
[[A_0, z], [z^T, t]] as a whole instead of setting t = z^T A_0^{-1} z after
rounding A_0, which keeps t at the solver's value when A_0 is nearly singular
(at 28 and degree 10 the difference is 0.007 in the bound).
At 28 and degree 10 the rounds after the first stopped improving: the raw
value stayed at 3.0913 and 3.0911 while the triple constraints failed by about
1e-3 between the samples (corrected 6.48 and 6.45). The solver meets its own
samples to 5e-7; the failures lie on the boundary of the Gram domain, the
triples of directions in one 2-plane through the centre, which the grid and
the random points reach only thinly. With `TCOPLANAR=n` every kind of triple
carries a fixed n-by-n grid of such coplanar triples, equally spaced in the
two angles and kept through every pruning, and the probes add a grid of twice
that density (`c28m_d10.log`). Its first round gave 3.09166 on the samples;
the triple violations between them fell from about 1e-3 to between 4e-5 and
4.4e-4, and the corrected value from 6.45 to 3.957. The worst violations now
sit inside the Gram domain, at triples with three nearly equal inner products
near 0.31, so the second round adds those points; a bound at 28 needs the
triple violations below about 2e-6. Round 2 gave 3.09216 on the samples and
3.227 corrected, round 3 3.09435 and 4.247: the violations move from one kind
to another, and the worst of rounds 1 to 3 sit at triples fixed by the
exchange of two points of the same type (BBB at (0.3316, 0.3316, 0.3316),
ABB at (-0.6188, -0.6188, 0.2783), AAB at (-0.6524, 0.3879, 0.3879)), where a
symmetric kernel has its critical points and from which the pruning between
rounds drops the samples once they are satisfied. With `TSYM=n` every kind
with two points of one type carries a fixed n-by-n grid on that symmetric
slice (u13 = u23 or u12 = u13, and for three equal types also the diagonal),
kept through every pruning, and the probes add one of twice that density
(`c28s_d10.log`, continued from the round-3 certificate). Its first round gave
3.09379 on the samples, but with the symmetric triples held the solver moved
weight onto the pairs and used the gaps between the pair samples, thinned to
5000 per kind by the pruning: the pair violations reached 0.56 and the
corrected value 134.9. The second round put those pair points back and gave
3.09513 (corrected 29.4); the run was stopped in round 3. Over the five rounds
at degree 10 the value on the samples went 3.09166, 3.09216, 3.09435, 3.09379,
3.09513, while the violations between samples moved from kind to kind and never
fell below about 1e-4. The sampled value bounds the optimum over all admissible
pairs and triples from below, and the margins of the exact check (2e-5 on each
of the 378 pairs and 2e-6 on each of the 3276 triples, as at 29 and 30) cost
about 0.014, so the exact check would need that optimum below about 3.089,
which the samples already exceed. The whole case at 28 is out of reach of the
kernel at degree 10.

The case at 28 does not split usefully at the count that limits the bound
at 29. With at least 19 of the centres within 2.05 (`case28_hiA.json`), the
kernel of three-point degree 6 gives 3.23113 in one round, the value of the
whole case at that degree (`c28hiA_d6.log`). At degree 10 the round-3
certificate is flat in the counts as well (`count_effect.py`,
`c28m_r3_counts.log`): its largest value over the count vectors, 3.09436, is at
22 centres within 2.0161, one in (2.05, 2.1], one in (2.2, 2.35] and four
beyond 2.35, and asking for 10 to 21 centres within 2.05, 24 within 2.2, 25 to
27 within 2.35, or at most 20 to 22 within 2.0161 changes it by at most
1.6e-4. Sharper radial counts would not lower the bound at 28; the obstacle
is the arrangement of 22 nearly touching centres with six farther out.

The case does split at the count of centres within 2.0161. Replacing the
condition N(2.0161) <= 24 of `case28_all.json` by N(2.0161) <= k
(`case28_lo{k}.json`) lowers the kernel steadily; in one round of three-point
degree 6 (floating point, `c28lo{k}_d6.log`):

| k | 24 (whole case) | 21 | 20 | 19 | 15 | 13 | 8 |
|---|---|---|---|---|---|---|---|
| value | 3.23113 | 3.21554 | 3.21021 | 3.20482 | 3.18288 | 3.17164 | 3.14307 |

At degree 8 the value drops by about 0.11 (k = 19: 3.09003 in round 1,
`c28lo19_d8.log`; k = 20: 3.10234, `c28lo20_d8.log`, stopped when the smaller
limits came in). `c28lo13_d8.log` ran k = 13 at degree 8 for four rounds: 3.06625, 3.07708, 3.07872, 3.07913 on the samples, corrected at best 3.14343 (round 3; in round 4 the pair functions bulged between samples); `c28lo13_d8b.log` continues it with the starting pair grid kept through every pruning (`KEEP_BASE=1`): round 1 gave 3.07929 on the samples and 3.09797 corrected, and the precheck of that certificate (`CM/c28lo13b_d8_r1.npz`, margins 5e-6, 5e-7, 1e-6, `BORDER=1`, `pre_c28lo13b_r1_small.log`) gives 3.100569 at (13, 0, 11, 0, 0, 0, 4), below 3.10330. Its first exact check (`CM/check_c28lo13b_r1.log`, 3.5 hours) proved the positivity, the six pair inequalities, the bins and nine of the ten triple inequalities, and failed on the kind FFF: a box with value 7.693729e-04 above the threshold 7.693708e-04. That threshold is the largest sampled value refined by SLSQP from the 40 best samples, plus margin3, and the sampling stopped below a narrow peak near 7.7176e-04 (ascent from 600 starts). The rerun raises that one threshold by 1e-5 (`TRIPLE_EXTRA=FFF:1e-5` in `multi_cap/combo_case_check.py`), which adds 4e-5 to the bound at (13, 0, 11, 0, 0, 0, 4), where four triples are of kind FFF, and passes (`CM/check_c28lo13b_r2.log`, copied to `multi_cap/runs/combo28_lo13_check.log`): 15 805 222 triple boxes and the largest bound 3.100609 at (13, 0, 11, 0, 0, 0, 4), below 3.10330. This was the first form of `prop:count28-few`, and `c28lo13b_d8_r1.npz` is `multi_cap/radial_certificates/combo28_lo13_d8.npz`. Over all count vectors the certificate reaches no further: with its proved thresholds the bound is 3.105310 at k = 14 and 3.149929 at k = 24, between 0.0043 and 0.0047 more for each further close centre (`CM/c28_kscan.py`, `CM/c28_kscan.log`). It does reach two more cases once the centres beyond 2.35 are bounded, since the worst vectors have four of them: at most 14 within 2.0161 with at most 3 beyond 2.35 (3.102452 at (14, 0, 11, 0, 0, 0, 3), `CM/case28_k14f3.json`) and at most 15 within 2.0161 with none beyond (3.101488 at (15, 0, 13, 0, 0, 0, 0), `CM/case28_k15f0.json`); at 15 with one beyond it would be 3.103335, just above. The pair, bin and triple inequalities do not involve the counts, so the passing check proves these cases once the exact largest bound over their count vectors lies below the level; the `EXTRA_CASES` option of `multi_cap/combo_case_check.py` computes it (`multi_cap/runs/combo28_extra_cases.log`). Narrower bins then reach one centre further. Three centres in the bins b, b' and b'' have inner products at most a(rho_b, rho_b'), a(rho_b, rho_b'') and a(rho_b', rho_b''), with rho_b the upper end of bin b, because a increases in each argument; so the inequality on triples of their kind holds on that smaller set with a smaller constant, and the same goes for pairs. The options `REFINE_TRIPLES` and `REFINE_PAIRS` of `multi_cap/combo_case_check.py` compute such constants with the same margins and check them by branch and bound; with `ONLY=NONE` the kinds themselves are not checked again (the passing check above covers them, with the same thresholds). For the triples in the bins (1, 3, 7) and (1, 6, 7) the constant drops from 3.2152e-4 to 3.1880e-4, for (3, 3, 3) from 5.4355e-4 to 5.3549e-4, for (1, 1, 3) from 5.6233e-4 to 5.6168e-4, and for the pairs in (3, 6) and (6, 6) from 6.59702e-2 to 6.59505e-2 and 6.59697e-2; the bins (1, 7, 7) and (1, 1, 6) give no measurable gain. The check (`multi_cap/runs/combo28_refined_check.log`, 13 315 362 further boxes, 8 792 068 of them for the combinations that lower a constant) gives 3.102386 at (14, 0, 9, 0, 0, 1, 4) over the 62 865 count vectors with at most 14 within 2.0161 (`CM/case28_k14.json`), and 3.102823 at (15, 0, 8, 0, 0, 3, 2) over the 55 611 with at most 15 within 2.0161 and at most 2 beyond 2.35 (`CM/case28_k15f2.json`), both below 3.10330; at 15 with 3 beyond it would be about 3.10479 at (15, 0, 8, 0, 0, 2, 3), above. Smaller margins then reach one case further on each side. The pair inequalities take the margin 1e-6, except the kind AA, which keeps 5e-6 because it comes within 1e-5 of its largest value all over the face of its domain where the two centres touch (`PAIR_MARGIN=AA:5e-6` in `multi_cap/combo_case_check.py`); the triples take 5e-8 (FFF still 1e-5 more) and the bins 1e-7. The full check with these margins and the refined bins (`multi_cap/runs/combo28_squeeze_check.log`: 3 532 617 pair boxes, 8 511 bin intervals, 17 006 005 triple boxes and 14 396 003 refined boxes) gives 3.099739 at (14, 0, 9, 0, 0, 1, 4) over the 62 865 count vectors with at most 14 within 2.0161, 3.102197 at (15, 0, 8, 0, 0, 2, 3) over the 62 063 with at most 15 within 2.0161 and at most 3 beyond 2.35 (`CM/case28_k15f3.json`), and 3.102468 at (16, 0, 7, 0, 0, 5, 0) over the 27 423 with at most 16 within 2.0161 and none beyond 2.35 (`CM/case28_k16f0.json`), all below 3.10330 (the floating-point precheck is `multi_cap/runs/combo28_squeeze_precheck.log`). This is `prop:count28-few`. With its thresholds the bound is 3.104750 at (15, 0, 8, 0, 0, 1, 4), with 4 beyond 2.35, and 3.104002 at (16, 0, 7, 0, 0, 4, 1), with 1 beyond, both above, and it rises to 3.114620 at 17 close and 3.148022 at 24 (`CM/c28_open_cases.py`, `CM/c28_open_cases.log`, floating point); narrower bins do not close that gap, so the next step needs a certificate built for more close centres. `CM/c28_config_test.py` checks the implementation on actual sets of 28 directions and distances, where the bound B(Y) of the kernels must lie above the union of caps U(Y): on the 12-close witness U = 2.263412 and B = 2.593852, and on the root system with four centres in deep holes U = 3.196365 and B = 3.702009 (`CM/c28_config_test.log`). A further round from those samples (`c28lo13_d8c.log`) gave 3.07941 and 3.10486 corrected and was stopped for the check.

At k = 15 the kernel of three-point degree 10 has more room on the samples.
`c28lo15_d10.log`, from the samples of round 2 of `c28s_d10.log`, gave 3.05709
on the samples and 3.92182 corrected, and its continuation `c28lo15b_d10.log`
gave 3.05883 and 3.15940, then 3.05807 and 3.40545; every round ended in a
numerical error of the solver. `c28lo15c_d10.log` continued it with a penalty
on the traces of the three-point blocks in the objective (`TRACE_REG=3e-4` in
`CM/combo_direct.py`), to damp the swings between samples, and was stopped in
its first round, slowed by the exact check beside it, to give its place to
degree 8 at k = 16 (`c28lo16_d8.log`, from the samples of the certificate of
`prop:count28-few`). Its first round gave 3.09593 on the samples (3.14856
corrected), 0.0165 above the certificate at k = 13; with the thresholds of
`multi_cap/runs/combo28_squeeze_precheck.log` that certificate bounds its case
0.0154 above its value on the samples, so a certificate for all of k = 16
would land near 3.111, and the run was stopped. `c28k15f4_d8.log` aimed instead
at the one case with 15 close that the check at smaller margins leaves, 15
within 2.0161 and 4 beyond 2.35 (`case28_k15f4.json`), where the certificate
of `prop:count28-few` gives 3.104750: its first round gave 3.08978 on the
samples. With the thresholds its solver found on its own samples, the
certificate of `prop:count28-few` already gives 3.07924 at k <= 13 (the
programme's 3.07929), 3.09042 on that case and 3.09607 at k <= 16
(`CM/c28_sampled_eval.py`, `CM/c28_sampled_eval.log`), so a certificate built
for either case gains at most about 6e-4 on the samples, less than the
1.45e-3 by which the level is missed. At degree 8 the kernel is saturated near
15 close centres. `c28lo16_d10.log` returned to degree 10 with the trace
penalty, at k = 16, from the samples of `c28lo15b_d10.log`; it was killed for
memory in its first round, with no round finished, when the mixing programme
below was started beside it.

A convex combination of certificates is again a certificate: the positivity
conditions are convex, and every inequality is linear in the kernel, with its
threshold taken for the combination. `CM/c28_hull.py` takes the certificates
at 28 with the same types (degrees 6, 8 and 10; a certificate of lower degree
enters with its three-point blocks embedded in the layout of degree 10, which
changes none of its functions) and finds the convex weights that minimise the
bound over a chosen set of count vectors. With only the weights unknown the
programme is a linear programme, so it takes about 3 million samples, and it
adds the local maxima of each combination, found by compass search, as
further samples until its thresholds stop moving. The certificates of degree
10 stopped on numerical errors of the solver and have eigenvalues near -1e-6
in their blocks. The exact check sets negative eigenvalues to zero before it
rounds, and for these certificates that moves the triple polynomials by up to
1e-5, so the programme projects every certificate onto the positive
semidefinite matrices first (without that step its first values, 3.08622 for
15 close with 4 beyond 2.35 and 3.09929 over the whole case, were not what
the check would see; the precheck of that mix gave 3.11286). With the
projection, in floating point and with the margins of the check, the best mix
gives 3.13159 over the whole case, 3.10618 at k = 17, 3.10076 over every
vector with k <= 16, 3.09916 over 15 close with 4 beyond and 16 close with 1
beyond, and 3.10014, 3.10039 and 3.10074 at 16 close with 2, 3 and 4 beyond.
The two mixes below the level that cover the open cases at k <= 16 are
`multi_cap/radial_certificates/combo28_mix1516_d10.npz` (15 close with 4
beyond, 16 close with at most 1 beyond) and
`multi_cap/radial_certificates/combo28_mix16_d10.npz` (k <= 16). Their
prechecks at margins 2e-6 (AA 5e-6), 2e-7 and 5e-7 give 3.098860 on
`CM/case28_k15f4.json` and 3.098060 on `CM/case28_k16f1.json`
(`multi_cap/runs/combo28_mix1516_precheck.log`), and 3.101264 at (16, 7, 1,
0, 0, 0, 4) on `CM/case28_lo16.json` (`multi_cap/runs/combo28_mix16_precheck.log`),
all below 3.10330. The precheck's threshold for the kind AAA lies 2e-7 below
a local maximum that the mixing programme found (and for AAB in the second
mix it has only 1e-8 to spare), so the exact checks raise those thresholds
(`TRIPLE_EXTRA=AAA:4e-7`, and `AAB:2e-7` in the second). The exact checks of
both mixes at three-point degree 10 are running
(`multi_cap/runs/combo28_mix1516_check.log`, `multi_cap/runs/combo28_mix16_check.log`);
until one of them passes, these cases stay open.
A certificate for some k reduces (C) at 28 to a statement about directions
alone: no 28 centres satisfying the radial counts of 28 (21 within 2.1, 22
within 2.15, 23 within 2.2, 24 within 2.35) have k + 1 of them within 2.0161.
Since a(r, s) increases in both distances, it suffices to rule out the
directions with every centre at the largest distance its count allows.
`CM/code_feas_dist.py` searches for such directions by local minimisation of
the squared violations (`CM/c28_direction_search.log`): with 2000 starts it
fits 9 centres within 2.0161 and none of 10 to 20 (least largest violation
1.6e-3 at 10, 2.8e-2 at 20). Random starts miss rare arrangements: moving one
centre of a 9-centre code in from 2.1 to 2.0161 in small steps
(`CM/c28_continue.py`, `CM/c28_chain.py`) gives a code with 10, and a search
that makes the 21 distances within 2.1 variables and rewards those at 2.0161
(`CM/c28_maxclose.py`) finds 12 in 2000 starts and never 13
(`CM/c28_close12_witness.txt`, slack 1e-5; from it `CM/c28_extend.py` misses
13 by 4.2e-3, and 600 steps of basin hopping, `CM/c28_hop.py`, stay at 12). At 27 the same search finds 18. So the limit k at 28 has to be
at least 12, and at 27 at least 18. A miss in
a local search is evidence, not a proof, and the statement is of the same kind
as the 24-plus-one exclusion below, which three-point bounds do not settle: the
typed three-point bound on 20 centres within 2.0161, one at 2.1, three at 2.35
and four at sqrt6 stays at 0 at degree 6, and so does the one on 23 centres
within 2.0161, one at 2.35 and four at sqrt6 at degrees 8 and 10 (`CM/c28_typed_dirs.py`).

## musin: where the two-point kernel fails

`dual_where.py` reads off the optimal dual of the two-point programme of
`thm:count31` (`multi_cap/radial_count_sdp.py`): the fictitious configuration
that the kernel cannot exclude. At 25 centres it puts all 25 at distance 2,
with pair inner products near 0.5, -0.15, -0.35 and -0.85; at 30 it puts 23.6
at distance 2 and 6.4 near sqrt6 (`dual_where.log`). The kernel is blind in the
way Delsarte's bound for the kissing number of R^4 (25.56) is blind.

`musin_scan.py` adds Musin's relaxation (Ann. of Math. 168, 2008): the pair
inequality is dropped for nearly antipodal pairs, and their excess is bounded
centre by centre by the number of centres that fit in a cap about the
antipode. At 30 centres the bound stays at 3.1441 for every threshold from
0.95 to 0.6 (`musin_scan_30.log`): the dual has no surplus near the antipode.

## localisation: the room for a budgeted localisation

Both open statements reduce to one statement (L): 24 centres with
T(Y) <= 8 + eta have directions within an explicit root-sum-square distance
rho(eta) of a root system. Through `cor:no-room`, the residual case at thirty
would have needed rho < 0.206 at eta = 0.00368; the kernel on typed triples
settled that case instead (`thm:count30`), and (L) remains the route for
twenty-five to twenty-seven centres, for the cases of twenty-eight that
`prop:count28-few` leaves (all with at least fifteen within 2.0161), and for
(G).

`budget_far.py` maximises the distance to the nearest root system under
T(Y) <= 8 + eta and the packing conditions, from root systems pushed out at
random (`budget_far.log`). The largest distance found is 0.0971 at
eta = 0.00368 (19 feasible ends of 24 starts) and 0.1291 at eta = 0.0334.
These are local searches near the root system, not bounds.

`room_table.py` gives, for one further centre at distance r, the budget
eta = S(r) and the distance that `cor:no-room` (ii) needs, with a(sqrt6, rho)
replaced by a(r, rho24) (`room_table.log`). `budget_far.py eta starts seed rmax`
holds the 24 centres within rmax. At the worst budget, eta = S(2.0161) = 0.127,
the searches reach 0.339, 0.378 and 0.378 for rmax = 2.25, 2.35 and 2.444
(`budget_far_r.log`), against 0.348, 0.288 and 0.231 needed: for a close
further centre the root-sum-square route does not suffice, and the hole form of
(L) is needed.

`hole_room.py` holds the 25th centre at a given distance r and minimises T
over 25 centres (`hole_room.log`): the least T found is 8.290, 8.287, 8.277,
8.259, 8.276, 8.275 and 8.264 at r = 2.0161, 2.05, 2.1, 2.2, 2.3, 2.4 and
2.449, so the hole form of (L) has a margin of about 0.26 at every distance.

`link_census.py` lists, point by point, the near neighbours (inner product at
least 0.4), the distance of the link from a cube and the Voronoi cell on S^3
(`link_census.log`). At the root system and at the ends of `budget_far.py`
every point has 8 near neighbours with a cube-like link; the second code has
none (6, 7 and 9 near neighbours).

`s3_cell.py` shows that a cell-by-cell volume bound on S^3 cannot prove (L):
with 8 neighbours at 60 degrees whose link is the square antiprism, the
Voronoi cell of a direction has volume at most about 0.817, below the
pi^2/12 = 0.8225 of the root system (`s3_cell.log`, Monte Carlo). A proof of
(L) along the lines of dimension three therefore has to move volume between
neighbouring cells.

## density: a bound for every packing

Section "A density bound from the cells alone" of the paper (`prop:levels`,
`thm:density-cells`) bounds the union of caps U(Y) for every count M from 24
to 30. At 25 and 27 to 30 the bound comes from two-point certificates with no
split; at M = 24 and M = 26 it comes from the combined kernel of `CM/`
(labelled pairs and typed triples), run against a level with
`multi_cap/combo_case_check.py`. At M = 26, degree 8, with LEVEL=3.3352 on
`CM/case26_all.json` and `multi_cap/radial_certificates/combo26_level_d8.npz`
(the round-3 certificate of `CM/c26h_d8.log`), it passes with
U(Y) <= 3.329050 (`multi_cap/runs/combo26_level_check.log`). At M = 24,
degree 6, with LEVEL=3.3291 on `CM/case24_all.json` and
`multi_cap/radial_certificates/combo24_level_d6.npz` (the round-1 certificate
of `CM/c24lvl_d6.log`, 3.21901 on its samples), it passes with
U(Y) <= 3.253872 (`multi_cap/runs/combo24_level_check.log`); dropping the
pair terms gives only 24 S(2) = 3.335124. At M = 25 the same kernel at degree
6 stays at 3.313 on its samples, and its round-1 certificate prechecks at
3.3768, above the two-point level 3.3274 (`CM/c25lvl_d6.log`). Its largest
value is at the bin counts (17, 0, 4, 1, 1, 2, 0), with exactly the seventeen
centres within 2.05 that `prop:C-radial` proves, as the level at 26 is
largest at its fifteen. So every cell has
volume at least 9 pi^2/8 - max L_M = 7.7742, the largest level being
L_26 = 3.3291, and every packing of unit balls in R^4 has density at most
0.63477. This is below
the three-point bound 0.63611 of Cohn, de Laat and Salmon, and above
pi^2/16 = 0.61685; it is a density bound, not statement (G) or (C).

| file | what it is |
|---|---|
| `levels.py` | floating point: the least level L that a certificate can reach for M centres, by bisection; used to choose the levels |
| `cert_M.log` | the certificate search, `multi_cap/radial_case_sdp.py M --level L`, writing `multi_cap/radial_certificates/density_M.json`; for M = 26, `cert_26_card.log`, with the spec `density_spec_26_card.json` (split at N(2.03), level 3.3503) |
| `check_M.log` | the exact check, `multi_cap/radial_case_check.py`: positivity by exact LDL^T, K <= Pi by branch and bound with Pi in Arb, the bin bounds, and the bound with no assumption on the count vector. All seven pass: two-point levels 3.3352, 3.3274, 3.3501, 3.3090, 3.2587, 3.2080, 3.1538 for M = 24 to 30 (`check_26_level3353.log` is the earlier check of M = 26 with no split, 3.3524); the paper takes 3.3291 at M = 26 from the combined kernel above |
| `case26_all.json`, `c26all_d2.log` | floating point: an early two-point kernel plus a typed three-point kernel at M = 26 with these coarse types and no counts, value 3.34979; superseded by `CM/case26_all.json` with the counts of `prop:C-radial`, which certifies 3.3291 |

Before the combined kernel, a bound below 0.63611 was sought with the pair
kernel alone, which needs L_26 <= 3.34549. The route tried there splits the
26-centre case by counts at 2.0161 and 2.03 (`split_level.py`, floating point):

| leaf | counts | level | what removes its complement |
|---|---|---|---|
| A | N(2.0161) <= 23, N(2.03) <= 25 | `split26_A.log` | N(2.03) >= 26: at most 25 points of S^3 with inner products at most 0.51468 (`card26_d10_*.log`) |
| B | N(2.0161) = 24, N(2.03) <= 24 | `split26_B.log` | N(2.03) >= 25: the typed exclusion at t2 = 0.5114 below |

With N(2.03) <= 25 left out, leaf B is too high: {N(2.0161) = 24, N(r2) <= 24}
gives 3.34717 at r2 = 2.0248 and 3.34708 at r2 = 2.025. Within leaf B the
optimum sits where a centre lies in (2.03, 2.05]: with N(2.05) <= 24 the level
is 3.33516 (`split26_B1.log`), with N(2.05) >= 25 it is 3.34471
(`split26_B2.log`), so an exclusion at r2 = 2.05 (t2 = 0.5163) would bring the
level to 3.34336, that of leaf A.

A cardinality bound alone does not move the two-point level: adding
N(2.0793) <= 24 at M = 25 (no 25 points of S^3 with inner products at most
0.5374, true numerically, the best 25-point code found having 0.537429,
`multi_cap/runs/code25_best.txt`) leaves the level at 3.33003
(`ladder25_t5374.log`).

The typed exclusion: 24 directions with inner products at most t1 = 0.508
(centres within 2.0161) leave no further direction with inner product at most
t2 with all of them (`multi_cap/typed_cardinality_sdp.py`; a corrected Z below
0 excludes the code in floating point, `multi_cap/typed_cardinality_check.py`
is the exact check). A centre within r2 of the origin has t2 = a(2.0161, r2):
0.5101 for r2 = 2.0248, 0.51135 for 2.03, 0.5163 for 2.05, 0.6141 for sqrt6.

| log | degree, t2 | result |
|---|---|---|
| `typed_d10_5101.log` | 10, 0.5101 | sampled Z +0.0255: no certificate of degree 10 (the sampled programme is a relaxation) |
| `typed_d14.log` | 14, 0.5114 | sampled -1.95, corrected +6.60 after one round (old sampling) |
| `typed2_d10_control.log` | 10, 0.508 | control, 25 points at 0.508, which `thm:kissing-stable` excludes: corrected Z -1.22 at round 2, so the refinement converges |
| `typed2_d12_5101.log` | 12, 0.5101 | corrected Z -1.26 at round 4: excluded in floating point |
| `typed2_d14_5101.log` | 14, 0.5101 | corrected Z +9.58 after two rounds |
| `typed2_d12_5114.log` | 12, 0.5114 | the exclusion leaf B needs: corrected Z 25.9, 16.7, 21.2, 0.506 in rounds 1 to 4 (sampled -0.97 to -0.42); the run was stopped by the memory limit in round 5 |
| `typed2_d12_5163.log` | 12, 0.5163 | sampled Z +0.011: no certificate of degree 12 |
| `typed2_d12_6141.log`, `typed2_d14_6141.log` | 12 and 14, 0.6141 | sampled Z +0.167 and +0.018: no certificate; an exclusion at sqrt6 would settle (C) when 24 centres lie within 2.0161 |
| `typed2_d12_071_control.log`, `typed2_d14_071_control.log` | 12 and 14, 0.71 | control: the deep holes of the 24-cell are at inner product 0.7071, so no certificate exists; sampled Z +0.951 and +0.192 |

Degrees 16 and 18 at t2 = 0.6141 are run on a larger machine
(`bigmachine/run_typed.py`): degree 16 gives sampled Z +0.0015 in each of
three rounds, so no certificate of degree 16 either. The sampled values at
sqrt6 fall about tenfold per degree (+1.87, +0.167, +0.018, +0.0015 at
degrees 10 to 16); those of the 0.71 control, where no certificate can
exist, fall too (+5.56, +0.951, +0.192 at degrees 10 to 14), so the fall
alone does not point towards a certificate.

A four-point kernel does not help at the degrees that fit here.
`C30/typed4pt.py` adds to the typed three-point kernel of `C30/typed3pt.py` a
kernel that fixes two of the directions, e1 and e2, expands every other
direction x in Chebyshev polynomials of (<e1, x>, <e2, x>) (and, with
`WITH_S=1`, of <e1, e2>), and multiplies by the zonal functions of the plane
orthogonal to e1 and e2; summed over the code it is a sum over pairs, triples
and quadruples, and `typed4pt.py selftest` checks that grouping against the
direct sum over the ordered poles (difference 1e-10). On 25 directions with
inner products at most t the three-point part of degree 8 alone excludes the
code up to t = 0.507 (corrected value -0.0096 there) and gives 0 from 0.508
on; at degree 4 it gives 0 already at 0.5. Four-point parts of degree 3 and 5
at t = 0.508 (over three-point degree 8), and of degree 3 (with or without
<e1, e2>) and 5 at t = 0.5 (over three-point degree 4), all settle at the
value 0 once the samples are refined. The negative values of their first
rounds, down to -31, came from gaps between the quadruple samples (0.035 to
0.053 in round 1), and fell to 0 as the gaps closed. On the 24-plus-one
exclusion at sqrt6 the four-point part of degree 3 gives 0 as well. Degree 5
with <e1, e2> needs more memory than this container has. Commands and output
are in `C30/typed4pt_runs.log`.

Local bounds on the number of neighbours do not help either.
`CM/local_counts.py` adds to the programme of `CM/combo_direct.py`, for each
pair of types (s, t) and each band u >= alpha of inner products of
directions, a bound kappa(s, t, alpha) on the number of centres of type t in
that band around a centre of type s: projected to the link sphere S^2 they are
separated, and the solved cases of the Tammes problem (up to 14 points; the
bound of Fejes Toth beyond) limit how many fit, for example 9 A-neighbours of
an A centre with u >= 0.45 and 7 A-neighbours of an F centre with u >= 0.6.
Each pair inequality is relaxed by multipliers on its bands, and the bound
pays n_s kappa for each. On the whole case at 28 and degree 6 the dual of the
programme describes the relaxed configuration: 24 centres of type A at
distance 2 with about 8.5 neighbours each at u = 0.5, which the count 9
allows, and four of type F at sqrt6, each with about 7.5 A-neighbours at the
largest inner product 0.612. Bands from 0.2 to 0.55 leave the optimum at
3.23114 (3.23113 without them); bands at 0.6 and 0.61 bind for the F centres
and lower it to 3.22891, while the relaxed configuration moves half a
neighbour to just below the band (`CM/local_counts_c28.log`). The near-D4
arrangements that defeat the kernels are locally possible; what excludes
them is global.

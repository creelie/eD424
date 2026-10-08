# multi_cap/

The contact and non-contact cases of sec:deviation-domain and sec:noncontact, file by file. The subdirectories `second_order/` (sec:second-order) and `stability/` (the corollary "No room beside a near root system" and the floating-point searches behind it) have their own README files.

## Scripts

  closure_lemmas.py
                              Supports sec:closure (lem:holes and lem:no-triple,
                              prop:inversion-hull, cor:root-subsets and cor:count,
                              and the angle quoted after thm:m23-labelled).
                              Symbolic checks of the identities,
                              integer enumeration of the orthogonal root
                              pairs at the vertices of the 24-cell and of
                              every set of at most four deleted roots, and
                              ball arithmetic for S(2), the bound
                              9 pi^2/8 - 22 S(2) > 8.046, the thresholds of
                              Psi beside those of Phi, and arccos(sqrt6/4).
                              A few seconds.  Log: runs/closure_lemmas.log.
                              The same exact content is in
                              lean/D4Closure.lean.

  labelled_certificate_check.py
                              Supports thm:m23-labelled (at most 23 centres
                              within sqrt 6).  Rebuilds the certificate of
                              thm:certificate and its bound exactly, rechecks
                              the positivity of its matrices, evaluates the
                              constants s(D), a_D, fr(1, 1, 1/2), r, kappa
                              and c in ball arithmetic (python-flint), reruns
                              the branch and bound of (C), and runs the two
                              branch and bounds of the ranges II_s and II_f
                              of t, with the tables of the Gamma_i in ball
                              arithmetic.  About six minutes.  Log:
                              runs/labelled_certificate_check.log.  Option
                              --skip-region-1 leaves out the rerun of (C).

  facet_bounds_probe.py [trials] [seed]
                              Supports step (b) of thm:near-contact.  Floating
                              point, exploration: on random and adversarial
                              perturbations of the root system, the facet
                              formula for the derivative of the volume
                              reproduces vol P(1) - 8, and the facet bounds of
                              step (b) hold (used to at most 82 and 14 per
                              cent over 4320 facets).  Log:
                              runs/facet_bounds_probe.log.

  explicit_eps0.py DATA      Supports thm:m24-robust and thm:near-contact (the explicit
                              epsilon_0 = 4e-26).  DATA is the folder
                              proofs/4_24 of the certificate in
                              ../third_party/llm24-certificate.  Bounds the
                              sum-of-squares terms of the certificate whose
                              weight can change sign (ball arithmetic,
                              python-flint), divides sigma_2 exactly by its
                              zeros and bounds the quotient below, checks
                              the minor bounds of lem:root-lattice-robust, and evaluates
                              the facet-by-facet and hull estimates of
                              thm:near-contact in exact arithmetic (the
                              neighbourhood sum of delta_i <= 4e-3).
                              About fifteen seconds.  Log:
                              runs/explicit_eps0.log.

  certify_cardinality.py CERT [e1] [e2] [wmin]
                              Supports thm:kissing-stable (the kissing number is
                              stable).  CERT is a file of
                              cardinality_certificates/.  Exact LDL^T of the
                              matrices F_k, the bound f(1) + F(1,1,1)
                              exactly, and the two interval branch and bounds
                              (on [-1, t] and on the ordered admissible
                              domain in [-1, t]^3) with the routines of
                              certificate_check.py, run up to the least
                              double at or above t.  About 80 seconds at
                              degree 8.  Logs: runs/certify_cardinality_*.log.

  cardinality_sdp.py d t [rounds] [bound]
  cardinality_sdp.py sweep d rounds t1 t2 ...
                              Finds the certificates of thm:kissing-stable (with a
                              bound, it fixes f(1) + F(1,1,1) and maximises
                              the least slack) and draws the curve of fig:positive-slack(a).  Floating point, sampled constraints
                              (cvxpy, Clarabel).  Logs:
                              runs/cardinality_search_*.log,
                              runs/cardinality_sweep_d*.log.

  robust_ceiling.py           Supports rem:robust-ceiling: the largest slack the
                              proof of thm:m24-robust can reach with the triple
                              and quadruple terms set to zero, from the exact
                              sigma_2.  Log: runs/robust_ceiling.log.

  code25_search.py [seed] [starts]
                              25 points of S^3 with largest inner product
                              0.53743 (minimal angle 57.49 degrees), by descent
                              on a smoothed maximum; writes
                              runs/code25_best.txt and rechecks its Gram
                              matrix.  Log: runs/code25_search.log.

  three_point_probes.py windows d kappa win [rounds]
  three_point_probes.py cap d tau [rounds]
                              Supports statements (i) to (iv) of sec:closure.
                              Floating point, sampled constraints,
                              exploration: whether a three-point certificate
                              can force the pair inner products of a 24-point
                              code into windows about -1, -1/2, 0, 1/2 (at
                              degree 12 it cannot, even with half-width 0.2),
                              and whether it can exclude 24 contacts plus one
                              centre with inner products up to tau (at degree
                              7: yes at tau = 1/2, no at tau = 0.612).  Log:
                              runs/three_point_probes.log.

  hole_labelled_sdp.py mode d kappa tau n [rounds]
                              Supports rem:what-c-needs.  Floating point, sampled
                              constraints refined in rounds, Clarabel.  mode
                              "hole": n directions of slack kappa and one
                              further direction at inner product at most tau
                              with each; mode "plain": the n directions alone.
                              Minimises the value Val(n) of a labelled
                              three-point certificate with c_AA = -1; a
                              positive value (nothing excluded) is robust to
                              the sampling.  With kappa = 0.008 and tau =
                              0.6141: Val > 0 at n = 24 (degrees 10, 12) and
                              n = 24.5 (degrees 10, 12); plain: Val > 0 at
                              n = 24.9 (degree 10) and n = 24.7 (degree 12).
                              Logs: runs/hole_labelled_*.log.

  cap_probe.py d tau kappa [rounds]
                              Supports statements (i) to (iv) of sec:closure.
                              Floating point, sampled constraints,
                              exploration: the cap probe above with the 24
                              directions at slack kappa, as thm:kissing-stable
                              allows for kappa = 0.008.  At degree 7, with
                              kappa = 0.008 and tau = 0.6141 (a further centre
                              within sqrt 6), the best certificate has value
                              0: no exclusion.  Log: runs/cap_probe_d7.log.

  radial_count_sdp.py M D r [eps [nd nu]]
  radial_count_sdp.py scan D r M1 M2 ...
                              thm:count31 and tab:kernel-limit.  Floating point,
                              sampled constraints: finds a two-point kernel,
                              polynomial of degree r in the distance and
                              of degree D in the inner product, whose
                              pair inequality bounds the union of the caps
                              of lem:no-triple by M m + t/2 with m < 0, and
                              writes it in exact dyadic form to
                              radial_certificates/radial_M.json.  With
                              "scan" it prints the best such bound for each
                              count.  Logs: runs/radial_count_sdp_31.log
                              (python3 radial_count_sdp.py 31 12 4 1e-5 31
                              80), runs/radial_count_scan.log.

  cardinality_tight.py cert.npz e1 e2 N
                              rem:cardinality-tight.  Floating point, from
                              cardinality_certificates/cert_d10_t0.50800.npz
                              (thm:kissing-stable, N = 25) and
                              cert_d10_t0.51468.npz (thm:twenty-six,
                              N = 26): B = 1 + f(1) + F(1,1,1), the largest
                              values of f + 3F(1,u,u) + 1 and of F and
                              where they occur (u = t, and the triple
                              (2t^2 - 1, t, t) of two touching pairs in a
                              plane), both polynomials at the inner
                              products of the root system, and how the
                              room of the final inequality is used.  Used
                              in no proof.  Logs:
                              runs/cardinality_tight_kissing.log,
                              runs/cardinality_tight_26.log.

  count31_tight.py            rem:count31-tight.  Floating point, from
                              radial_certificates/radial_31.json: the
                              affine bound M m + t/2, the distances where
                              condition (c) is tight (the ends 2 and sqrt 6)
                              and the inner products where condition (a) is
                              tight for the end distances.  Used in no
                              proof.  Log: runs/count31_tight.log.

  combo_tight.py count29|count30
                              rem:typed-tight.  Floating point, from
                              combo29_d8.npz and combo30_d8.npz and the
                              thresholds their checks proved: the bound
                              split into centres, t/2, pairs and triples
                              at the count vectors where it is largest,
                              the count constraints active there (at 29,
                              N(2.05) >= 19; with 18 the bound is 3.1082),
                              the largest bound for each count within
                              2.05, the spread of f over the bins, and
                              where the pair and triple inequalities come
                              closest to their thresholds.  Used in no
                              proof.  Logs: runs/combo29_tight.log,
                              runs/combo30_tight.log.  With level26 the
                              same for the certificate of prop:levels at
                              M = 26 (largest 3.32906, at 15 centres
                              within 2.05; with 14 it would be 3.33156,
                              and with 23 centres within 2.444, 3.3590):
                              runs/combo26_level_tight.log.

  radial_count_check.py radial_certificates/radial_31.json
                              Proves thm:count31 from the certificate
                              alone: exact LDL^T for the positivity and the
                              Schur complement, a Bernstein branch and
                              bound of the pair inequality with the pair
                              term (lem:pair-closed) in Arb ball arithmetic at
                              the corners of 191 740 boxes, the per-point
                              bracket on 356 intervals, and
                              31 m + t/2 = 3.08537 < 9 pi^2/8 - 8.  Under
                              two minutes.  Log:
                              runs/radial_count_check_31.log.

  radial_case_sdp.py M --spec radial_certificates/case_M_spec.json
                              thm:count30: two-point certificates for (C)
                              at one count M, one for each case of a tree
                              of bounds on the numbers N(r) of centres
                              within r (the tree for M = 30 is
                              case_30_spec.json; its fifth leaf, marked
                              residual, is settled by combo30_check.py
                              below).  Floating point;
                              writes radial_certificates/case_M.json.
                              Without --spec it grows a tree by trying
                              splits.  A spec with "antipodal": true also
                              refines the samples along u = -1, where the
                              caps are disjoint, and "margins" sets the
                              margins tried, each a number or a pair (near,
                              far).  Log:
                              runs/radial_case_sdp_30.log.

  radial_case_check.py radial_certificates/case_M.json
                              Proves thm:count30 from the tree and the
                              certificates: the splits cover every packing,
                              exact LDL^T, the pair branch and bound of
                              radial_count_check.py, the bin bounds, and
                              the largest bound over the integer count
                              vectors of each case by exact dynamic
                              programming, against 9 pi^2/8 - 8 in Arb:
                              3.0617, 3.0992, 3.0968, 3.0845 for the four
                              certified cases at M = 30; 185 million boxes,
                              66 minutes on four cores.  Log:
                              runs/radial_case_check_30.log.

  radial_certificates/case_24_r2.0161.json, case_24_r2.05.json,
  case_24_r2.1.json, case_24_r2.15.json
                              prop:G-radial: the same certificates at
                              M = 24, one split each: N(rho) <= c - 1
                              against N(rho) >= c for (rho, c) =
                              (2.0161, 11), (2.05, 20), (2.1, 22) and
                              (2.15, 23), with the specs case_24_r*_spec.json
                              (those for 2.0161 and 2.05 keep the margins
                              3e-5 and 5e-5 where the caps are disjoint).
                              radial_case_check.py proves 3.10018, 3.09836,
                              3.07883 and 3.09251 < 9 pi^2/8 - 8, under a
                              minute each.  Logs: runs/radial_case_sdp_24_r*.log
                              and runs/radial_case_check_24_r*.log.

  radial_certificates/case_M_rRHO.json, M = 25..30
                              prop:C-radial: the same single splits at
                              M = 25 to 30 centres, one file per radius
                              rho and count M, with the specs
                              case_M_rRHO_spec.json.  Each proves that the
                              case N(rho) <= n - 1 has T(Y) > 8, so that
                              at least n of the M centres lie within rho
                              (tab:C-radial); case_30_r2.25.json,
                              case_29_r2.3.json and case_2{7,8}_r2.35.json
                              prove that 24 of the M centres lie within
                              2.25, 2.3 and 2.35.  The margins
                              are chosen per case, since the margin is
                              given up at each of the M(M - 1)/2 pairs.
                              Logs: runs/radial_case_sdp_M_r*.log and
                              runs/radial_case_check_M_r*.log.

  radial_certificates/density_M.json, M = 24..30
                              prop:levels and thm:density-cells: the same
                              certificates with no split, against a level
                              L instead of 9 pi^2/8 - 8 (radial_case_sdp.py
                              M --level L; the file keeps the level).  With
                              a level, radial_case_check.py also prints the
                              largest bound with no assumption on the count
                              vector, which holds for every set of M
                              centres since the multiplier and the bin
                              bounds are nonnegative: 3.3352, 3.3274,
                              3.3501, 3.3090, 3.2587, 3.2080, 3.1538 for
                              M = 24 to 30, the values L_M of tab:levels
                              except at M = 24 and 26, where the table takes
                              3.2539 and 3.3291 from combo_case_check.py
                              below.  density_spec_26_card.json splits M = 26
                              at N(2.03) <= 25, the other case being empty
                              by thm:twenty-six, and sets the small margins
                              that M = 26 needs, since the margin is given
                              up at 325 pairs.  Logs:
                              gap_closure/density/cert_M.log and
                              check_M.log.

  combo30_check.py radial_certificates/combo30_d8.npz 8 2e-5 5e-6 2e-6
                              thm:count30, fifth case (24 centres within
                              2.25, 6 beyond 2.4): the two-point kernel
                              labelled by distance plus the three-point
                              kernel on directions typed by distance
                              (lem:typed-triples), with the certificate of
                              gap_closure/C30/combo30p.py.  Exact LDL^T, the
                              six pair inequalities by tensor Bernstein
                              bounds with Pi in Arb (47 646 boxes), the bins
                              (4131 intervals), the nine triple inequalities
                              by second-order Taylor forms (8 275 056
                              boxes), and the bound at the five count
                              vectors: largest 3.087203 at (22, 1, 1, 0, 6),
                              below 9 pi^2/8 - 8.  115 minutes.  Log:
                              runs/combo30_check.log.

  combo_case_check.py case.json cert.npz d3 [m2 m3 mm]
                              The same check for a case file of
                              gap_closure/CM (statement (C) at 25 to 29
                              centres, the count constraints of
                              prop:C-radial), with the bound taken over
                              every allowed vector of bin counts in exact
                              arithmetic.  thm:count29: on
                              radial_certificates/combo29_d8.npz with
                              gap_closure/CM/case29_all.json and margins
                              5e-6, 1e-6, 1e-6 it passes: the six pair
                              inequalities (21 895 boxes), the bins
                              (17 762 intervals), the ten triple
                              inequalities (16 301 757 boxes), and the
                              largest bound over the 23 396 count vectors,
                              3.101462 at (19, 0, 4, 0, 0, 5, 1), below
                              9 pi^2/8 - 8.  104 minutes on four cores.
                              Log: runs/combo29_check.log.  With
                              LEVEL=L in the environment the last
                              comparison is with L instead, for a density
                              level: the case constraints hold whenever
                              U(Y) >= 9 pi^2/8 - 8, so a pass gives
                              U(Y) < max(9 pi^2/8 - 8, L) for every set of
                              M centres.  prop:levels at M = 26: on
                              radial_certificates/combo26_level_d8.npz
                              with case26_all.json, margins 2e-5, 2e-6,
                              2e-6 and LEVEL=3.3352 it passes: the six
                              pair inequalities (9 996 boxes), the bins
                              (6 323 intervals), the nine triple
                              inequalities (7 512 629 boxes), and the
                              largest bound over the 17 845 count vectors,
                              3.329050 at (15, 0, 9, 0, 0, 0, 2), below
                              3.3352; this is L_26 = 3.3291 of tab:levels.
                              46 minutes on four cores.  Log:
                              runs/combo26_level_check.log.  prop:levels
                              at M = 24: on
                              radial_certificates/combo24_level_d6.npz
                              (three-point degree 6) with case24_all.json,
                              the same margins, BORDER=1 and LEVEL=3.3291
                              it passes: the six pair inequalities (3 361
                              boxes), the bins (285 intervals), the seven
                              triple inequalities (104 028 boxes), and the
                              largest bound over the 376 count vectors,
                              3.253872 at (24, 0, 0, 0, 0, 0, 0); this is
                              L_24 = 3.2539.  Four minutes.  Log:
                              runs/combo24_level_check.log.
                              prop:count28-few, first at larger margins: on
                              radial_certificates/combo28_lo13_d8.npz with
                              gap_closure/CM/case28_lo13.json (at most 13
                              centres within 2.0161), margins 5e-6, 5e-7,
                              1e-6, BORDER=1 and TRIPLE_EXTRA=FFF:1e-5 it
                              passes: the six pair inequalities (3 279 356
                              boxes), the bins (3 698 intervals), the ten
                              triple inequalities (15 805 222 boxes), and
                              the largest bound over the 60 214 count
                              vectors, 3.100609 at (13, 0, 11, 0, 0, 0, 4),
                              below 9 pi^2/8 - 8.  3.8 hours on four cores.
                              Log: runs/combo28_lo13_check.log.  With
                              TRIPLE_EXTRA=FFF:1e-5 the threshold of the
                              triple kind FFF is raised by 1e-5 beyond its
                              sampled and refined maximum: without it the
                              first run passed everything else and failed on
                              FFF, whose narrow peak (about 7.7176e-4) the
                              samples missed by 3e-6 (log
                              gap_closure/CM/check_c28lo13b_r1.log).  With
                              ONLY=FFF (or a list of kinds) only those
                              triple branch and bounds run, from the same
                              thresholds, and the run ends PARTIAL.
                              Two further cases, now inside those of the next
                              paragraph:
                              with PRECHECK=1, the same certificate, case,
                              arguments and options, and
                              EXTRA_CASES=case28_k14f3.json,case28_k15f0.json
                              (gap_closure/CM; at most 14 within 2.0161 and
                              at most 3 beyond 2.35, at most 15 within
                              2.0161 and none beyond 2.35), the run
                              reproduces the thresholds of the passing check
                              and compares the exact largest bound over the
                              count vectors of each listed case with
                              9 pi^2/8 - 8: 3.102452 at (14, 0, 11, 0, 0, 0,
                              3) over 59 799 vectors and 3.101488 at (15, 0,
                              13, 0, 0, 0, 0) over 26 554.  The pair, bin and
                              triple inequalities do not involve the count
                              vectors, so the passing check covers them.
                              Log: runs/combo28_extra_cases.log.
                              Then, at the same margins, the cases with 14
                              and 15 within 2.0161: the same certificate, case,
                              arguments and options, with ONLY=NONE,
                              REFINE_TRIPLES=1-3-7,3-3-3,1-1-3,1-6-7,1-7-7,1-1-6,
                              REFINE_PAIRS=3-6,6-6 and
                              EXTRA_CASES=case28_k14.json,case28_k15f2.json
                              (at most 14 within 2.0161; at most 15 within
                              2.0161 and at most 2 beyond 2.35).  Each listed
                              combination of bins gets the constant of its
                              smaller set of inner products, checked by its
                              own branch and bound (13 315 362 boxes in all),
                              and the bound counts each pair or triple with
                              the constant of its bins where one is given:
                              3.102386 at (14, 0, 9, 0, 0, 1, 4) over 62 865
                              vectors and 3.102823 at (15, 0, 8, 0, 0, 3, 2)
                              over 55 611.  Log:
                              runs/combo28_refined_check.log.
                              The proposition as it stands: the same
                              certificate and case at margins 1e-6, 5e-8, 1e-7
                              with PAIR_MARGIN=AA:5e-6 (the AA pair inequality
                              is within 1e-5 of its largest value all over its
                              touching face), BORDER=1, TRIPLE_EXTRA=FFF:1e-5,
                              the REFINE_TRIPLES and REFINE_PAIRS above, all
                              kinds checked again, and
                              EXTRA_CASES=case28_k14.json,case28_k15f3.json,case28_k16f0.json
                              (at most 15 within 2.0161 and at most 3 beyond
                              2.35; at most 16 within 2.0161 and none beyond).
                              It passes: the six pair inequalities (3 532 617
                              boxes), the bins (8 511 intervals), the ten
                              triple inequalities (17 006 005 boxes), the
                              refined combinations (14 396 003 boxes), and
                              3.099739 at (14, 0, 9, 0, 0, 1, 4) over 62 865
                              vectors, 3.102197 at (15, 0, 8, 0, 0, 2, 3) over
                              62 063 and 3.102468 at (16, 0, 7, 0, 0, 5, 0)
                              over 27 423.  6.8 hours on four cores, beside
                              other runs.
                              Log: runs/combo28_squeeze_check.log.
                              Mixed certificates for the cases it leaves
                              (gap_closure/CM/c28_hull.py; convex
                              combinations of earlier certificates, each
                              projected onto the positive semidefinite
                              matrices, three-point degree 10):
                              radial_certificates/combo28_mix1516_d10.npz on
                              case28_k15f4.json with
                              EXTRA_CASES=case28_k16f1.json, and
                              radial_certificates/combo28_mix16_d10.npz on
                              case28_lo16.json, margins 2e-6, 2e-7, 5e-7,
                              BORDER=1, PAIR_MARGIN=AA:5e-6,
                              TRIPLE_EXTRA=AAA:4e-7 (and AAB:2e-7 for the
                              second).  Prechecks 3.098860 and 3.098060
                              (runs/combo28_mix1516_precheck.log) and
                              3.101264 (runs/combo28_mix16_precheck.log);
                              the exact checks are running
                              (runs/combo28_mix1516_check.log,
                              runs/combo28_mix16_check.log), and nothing
                              rests on them until one passes.

  pair_bb_fast.py             The branch and bound of radial_count_check.py
                              for K <= Pi, processed in numpy batches: the
                              same boxes, tests and counts, four times
                              faster; used by radial_case_check.py.

  pushout_check.py
                              thm:G-pushout: T on the one-centre ray at
                              delta = 0.1971 exceeds 8 (Arb), the pyramid
                              condition (1 + eta)^2 - 1 < 1/3, and
                              g(eta)/eta >= 2/5 on [0, 0.09855] (exact); and a
                              floating-point comparison of the bound
                              8 + sum(delta)/5 with qhull volumes of random
                              push-outs.  Log: runs/pushout_check.log.

  twentyfour_close_check.py
                              lem:twentyfour-close: 22 S(2) + 11 S(2.444) < 9 pi^2/8 - 8,
                              s(D) < B, the room 2.6492e-5 of thm:m23-labelled against (M - 23) S(2.444) for
                              M = 25..33, and 6 S(2.4) < 0.00368.  Arb at
                              200 bits, under a second.

  count_core_survey.py M K [starts] [seed]
                              rem:what-c-needs.  Floating point, exploration: the
                              least T over packings of M centres within
                              sqrt 6 with K of them held within rho_M, the
                              radius inside which at least 23 centres must
                              lie when T <= 8 (2.2677 for M = 25, 2.3273 for
                              M = 30).  With K = 23: 8.353, 8.472, 8.681,
                              8.781, 8.771, 8.937 for M = 25..30.  Log:
                              runs/count_core_survey.log.

  truncated_search.py M [starts] [seed]
  truncated_search.py rays
                              Supports statements (i) to (iv) of sec:closure.
                              Floating point, exploration: minimises the
                              right side of lem:no-triple over configurations
                              of exactly M centres within sqrt 6.  Logs:
                              runs/truncated_search_M24.log to _M27.log.
                              With "rays" it prints where that right side
                              reaches 8 on two families through the root
                              system (sum of delta_i = 0.155 pushed out
                              evenly, delta = 0.197 for one centre).  Log:
                              runs/truncated_search_rays.log.

  near_contact_probe.py
                              Supports the remarks after thm:near-contact.
                              Floating point, exploration: the inversion
                              hull beside the volume, and 200 volume
                              minimisations near the root system.  Log:
                              runs/near_contact_probe_200_seed5.log.

  shell_reduction.py
                              Supports sec:noncontact (prop:distance-criterion,
                              lem:24-no-shell, thm:local-general, tab:distance-thresholds). The
                              covering bound with every centre within
                              2 sqrt 2 counted by the cap it cuts, as a
                              function Phi of the distances alone, stopped
                              at 45 degrees so that no centre beyond
                              2 sqrt 2 enters.  Checks in ball arithmetic
                              (python-flint): the covering radius of the
                              root directions is 45 degrees; Phi > 8.044
                              for m <= 22 contacts; the thresholds of
                              tab:distance-thresholds; two examples showing why lem:shell-new and lem:radial-new need their hypotheses.  About 50 s.
                              Log: runs/shell_reduction.log.

  shell_neighbour_search.py
                              Supports the paragraph "The open case,
                              searched" of sec:noncontact and fig:noncontact(c).
                              Minimises the exact volume of the cell, with
                              its exact gradient, over configurations of
                              22 to 26 centres under the packing
                              constraints (SLSQP); optionally one
                              neighbour held at distance 2 + delta.
                              Floating point: exploration, not proof.
                              Logs: runs/shell_neighbour_search_*.log.

  root_deletions_exact.py
                              Supports prop:meet22 and prop:meet21.
                              Exact integer vertex enumeration of the cell
                              left when j = 1, 2, 3 roots are removed from
                              D_4. Scaling the roots to integer vectors of
                              squared length 2 makes every vertex of the
                              cell rational, of the form m/e with m
                              integral and e a positive integer, so the
                              circumradius test is the integer comparison
                              |m|^2 <= 2 e^2 and the test for a direction
                              that can be added is |m|^2 = 2 e^2.
                              Reports, over all 24, 276 and 2024 cases:
                              circumradius exactly 2 with the removed roots
                              as the only attaining directions, except at
                              the 96 triples pairwise at 60 degrees, where
                              the circumradius is sqrt6.
                              No floating point anywhere.

  cell600_exact.py
                              Establishes prop:cell600. The 120
                              vertices of the 600-cell with doubled
                              coordinates in Z[phi], every inner product
                              computed exactly in Z[phi]; two vertices are
                              closer than 60 degrees exactly when joined by
                              an edge (36 degrees), so the subsets with all
                              inner products at most 1/2 are the
                              independent sets of a 12-regular graph. The
                              script finds the 25 inscribed 24-cells as the
                              24-cliques of the graph of root-system
                              angles, then enumerates completely the
                              independent sets through a fixed vertex:
                              none of size 25, five of size 24 (the cells
                              through that vertex), and 115 of size 23,
                              every one inside a cell, so 115 x 120 / 23 =
                              600 = 25 x 24 in all. Writes the graph and
                              the cell list to cell600_graph.txt. Runtime
                              about ninety seconds; no floating point
                              anywhere. Run with --sat for an additional
                              SAT cross-check through python-sat (slow).

  cell600_enum.c
                              Independent C implementation of the size-23
                              enumeration of cell600_exact.py, on 64-bit
                              bitsets, reading cell600_graph.txt:
                                cc -O2 -o cell600_enum cell600_enum.c
                                ./cell600_enum
                              Prints 115, 600 and 0 and PASS in under two
                              seconds. The Lean project lean/cell600/ is
                              the third implementation.

  cell600.py
                              The sampling that preceded the enumeration:
                              two million maximal independent sets of the
                              same graph drawn by greedy growth along
                              random orders, sizes reported (10 to 22 and
                              24, never 23). Kept for the record;
                              superseded by cell600_exact.py.

  inradius_search.py
                              Exploration of 23-point codes, not cited in
                              the paper. Maximises the inradius
                              g(W) of conv(W), the cosine of the covering
                              radius, over 23-point configurations with all
                              inner products at most 1/2, by a trust-region
                              sequential linear programme with analytic
                              gradients: the facet offsets of the convex
                              hull and the pairwise inner products are
                              linearised in a tangent move of bounded size,
                              the LP maximising the smallest linearised
                              offset is solved (HiGHS through scipy), and
                              the move is kept only if the true inradius,
                              read from the facet equations of a fresh
                              hull, improves without any contact constraint
                              being violated. Three kinds of start in
                              rotation: random points brought to
                              feasibility through a relaxed-then-tightened
                              schedule; perturbed deletions; twenty roots
                              plus three random directions. Reports every
                              feasible endpoint's g and the multiset of its
                              inner products, and flags any g > 1/2.
                              Exploration, double precision throughout.
                                python inradius_search.py [starts] [seed]

  octahedral48_exact.py
                              The 48 unit quaternions of the binary
                              octahedral group (two 24-cells in dual
                              position), coordinates doubled in Z[sqrt2];
                              enumerates every subset of size 23 and 24
                              with inner products at most 1/2 and finds
                              the two root systems and their 48 deletions
                              only. Exact; a few seconds.

  three_point_reduction.py
                              Supports sec:strict-inequality and the r_* forms of
                              prop:second-order, prop:pair-budget and prop:two-point-barrier. Part 1:
                              the pair-only truncated-volume bound at the
                              limits r_23 and r_* (bracket, weight of a
                              60-degree pair, value at a deletion, pairs
                              needed for 8). Part 2: the three-point bound
                              at r_4 = arcsin sqrt(3/8), with the triple
                              cap measure by Monte Carlo, at a deletion
                              (8.140848) and at the root system
                              (7.968684). Part 3: the pair-angle linear
                              relaxation of the bound for both weights
                              (0.04157 of 0.08327 at r_23; 0.07380 of
                              0.09286 at r_*) and the minimising measure.
                              Under a minute (the Monte Carlo sample is seeded).

  truncated_volume.py
                              Supports sec:strict-inequality. Evaluates the volume
                              of the cell inside a ball of radius R by a
                              fixed quasi-random quadrature on S^3
                              (deterministic, 400000 points), checks it
                              against the exact volumes at the root system
                              and a deletion, then minimises it over
                              23-point configurations with inner products
                              at most 1/2 + delta on a decreasing schedule
                              of delta, by the trust-region sequential LP
                              of inradius_search.py with the analytic
                              gradient of the quadrature. Prints, per
                              level, the least truncated volume found, the
                              exact volume of that configuration, and the
                              number of endpoints. Exploration.
                                python truncated_volume.py [starts] [seed] [points] [R]
                              (default R = sqrt(8/5); R = 1.224744871391589
                              for sqrt(3/2)). About two hours per run.

  three_point_sdp.py
                              Supports sec:certificate: the three-point
                              (Bachoc-Vallentin) relaxation of the pair
                              inequality of thm:strict-reduction, and the
                              certificate of thm:certificate. Builds the
                              Gegenbauer polynomials of S^3, the matrices
                              Y_k for n = 4 (Legendre polynomials, in the
                              Chebyshev basis T_i(u) T_j(v)) and their
                              symmetrisation, imposes the condition (C) of
                              lem:certificate on a grid of admissible triples
                              and solves the semidefinite programme with
                              cvxpy and Clarabel. First mode: maximise the
                              bound, check on about 1.15 million further
                              triples, add the worst 3000 and repeat;
                              reports the bound reduced by the largest
                              violation (degree 6: 0.09011, degree 8:
                              0.09523, against the target 0.09286; the
                              two-point programme alongside: 0.0712).
                              Second mode (fifth argument): fix the bound
                              and maximise the least slack; writes
                              continuation_out/certificate_d8.npz (and a
                              plain-text copy, certificate_d8.txt, with
                              every number printed exactly). Needs cvxpy
                              and Clarabel (pip install cvxpy clarabel).
                              One to three minutes per solve, a quarter
                              of an hour for five rounds.
                                python three_point_sdp.py 8 30 CLARABEL 5
                                python three_point_sdp.py 8 30 CLARABEL 5 0.0929

  certificate_check.py
                              The proof of thm:certificate: verifies the
                              certificate in exact rational and interval
                              arithmetic, sharing no code with the solver.
                              Step 1, exact LDL^T of the nine matrices and
                              f_k >= 0; step 2, the bound computed exactly
                              against 8 - A_* from the closed form of A_*
                              in mpmath interval arithmetic; step 3, the
                              polynomial P expanded exactly (449 monomials,
                              degree 16, symmetric), omega and its two
                              derivatives tabulated from closed forms in
                              interval arithmetic, and the inequality (C)
                              verified on the ordered admissible domain by
                              an interval branch and bound with the
                              second-order Taylor form on each box (313780
                              boxes verified, 13517 discarded, 44 levels,
                              about six minutes). Stops at the first
                              failure with the box or the counterexample.
                              Needs numpy, mpmath, sympy.
                                python certificate_check.py 8 1e-6

  certificate_tight.py
                              rem:certificate-found and fig:certpair.
                              Floating point, from
                              continuation_out/certificate_d8.npz: the two
                              parts of the bound B (0.2345077 from f_0 and
                              -0.1416077 from F(1,1,1); f_1 + ... + f_8 is
                              5.2e-8), the least slack of (C) on the whole
                              domain, 2.30e-6 at the coplanar triple
                              (-sqrt3/2, -sqrt3/2, 1/2), which the samples
                              of the solver did not reach (their least is
                              1.3e-5), the slack at the triples of the root
                              system less one root, and its pair sum
                              0.127196.  Used in no proof.  Log:
                              runs/certificate_tight.log.

  root_lattices_rank4.py
                              Supports lem:root-lattice, the combinatorial half
                              of the twenty-four-point classification.
                              Enumerates every positive definite Gram
                              matrix with 2 on the diagonal and -1, 0, 1
                              off it of order 1 to 4 (1, 3, 23, 393 of
                              them), counts the integer solutions of
                              x^T G x = 2 within the rigorous box
                              |x_i| <= 15 by exact integer arithmetic,
                              and reports the largest counts 2, 6, 12, 24,
                              the count 24 occurring only at determinant
                              4 with the neighbour profile of D_4 (next:
                              20 at determinant 5, A_4). About a minute.
                              Log: runs/root_lattices_rank4.log.
                                python root_lattices_rank4.py

  llm24_certificate_check.py
                              Supports thm:m24, prop:verified and
                              sec:certificate-checked: an independent verification,
                              sharing no code with the authors' Julia
                              package, of the certificate of de Laat,
                              Leijenhorst and de Muinck Keizer (data set
                              doi:10.4121/74ce1c25-6fca-4680-8a36-e9c18e7e9594,
                              LasserreSphericalCodes.zip, md5
                              02acd5270f7b3fa799abdeb5291706fd, redistributed
                              in third_party/llm24-certificate; pass the
                              path of its proofs/4_24 folder). Of the seven steps of
                              their verification it repeats five: the
                              reading and format of the data (127 blocks
                              of total dimension 3726, the block of every
                              signature having exactly as many rows as
                              there are admissible tuples); positive
                              definiteness of all 127 blocks by Cholesky
                              in ball arithmetic at 256 bits (python-flint
                              arb, every pivot a ball inside the positive
                              reals; least pivot about 1.38e-15), the 81
                              blocks of size at most 16 also by exact
                              rational LDL^T; the structure of all 125
                              sum-of-squares prefactors (each a
                              nonnegative multiple of a domain weight);
                              the objective K(empty, empty) = 24 exactly;
                              and the two-point polynomial p_2, degree
                              16, computed exactly from the data and
                              shown by a Sturm sequence over Q to vanish
                              on [-1, 1/2] exactly at -1, -1/2, 0, 1/2
                              (multiplicities 1, 2, 2, 1). Steps 3 and 5,
                              the construction of the zonal matrices and
                              the polynomial identities that use them,
                              are done in zonal/ (see zonal/README.md).
                              Writes
                              llm24_out/llm24_p2.txt, from which
                              lean/gen_innerproducts_lean.py generates
                              lean/D4InnerProducts.lean. Needs
                              python-flint. About eight minutes.
                              Log: runs/llm24_certificate_check.log.
                                python llm24_certificate_check.py /path/to/LasserreSphericalCodes/proofs/4_24

  run_llm24_full_verification.sh
                              The computation this package does not
                              contain: on a machine with 128 GB of
                              memory and 8 cores, installs Julia 1.10,
                              runs the authors' complete verification
                              (zonal matrices and the polynomial
                              identities included, about three days),
                              then llm24_certificate_check.py on the
                              same data, and keeps both logs. Needs
                              LasserreSphericalCodes.zip in the current
                              directory.
                                bash run_llm24_full_verification.sh

  symmetric_search.py
                              Exploration of 23-point codes with a
                              symmetry, not cited in the paper. For every
                              rotation type of order
                              n <= 12 (angles 2 pi a/n, 2 pi b/n in two
                              orthogonal planes) and the two improper
                              involutions, lists every orbit structure
                              with 23 points in all (73 structures) and
                              runs the slack continuation inside the
                              class, the orbit representatives being the
                              parameters, from a number of random starts;
                              reports per structure the feasible
                              endpoints at slack 0, the best inradius,
                              the best at slack 0.01 and whether the best
                              endpoint is a deletion. Exploration; about
                              an hour for 16 starts per structure.
                                python symmetric_search.py [starts] [seed]

  runs/                       The recorded output of the runs quoted in
                              the paper: cell600_exact.log and
                              cell600_enum.log (the enumeration, Python
                              and C), three passes of
                              slack_continuation.py (seeds 1, 2, 3 with
                              40, 80 and 60 fresh starts per level; the
                              largest inradius at each level over the
                              three is the value the exploration reports),
                              and
                              inradius_search_300_seed7.log (300 direct
                              maximisations at slack 0: 94 feasible
                              endpoints, all deletions), and
                              symmetric_search_16_seed1.log,
                              three_point_reduction.log, the two
                              truncated_volume runs (truncated_volume_r4.log
                              and truncated_volume_r3.log), the relaxation
                              runs three_point_sdp_d6.log and
                              three_point_sdp_d8.log, the certificate run
                              three_point_sdp_certificate_d8.log, its
                              verification certificate_check_d8.log, the
                              rejected altered certificate
                              certificate_check_altered.log (f_0 raised by
                              1e-5), the Lean check
                              D4Certificate_lean.log, the build log of
                              the Lean branch and bound
                              lean_certificate_build.log and its counter
                              run lean_certificate_stat.log (421881
                              boxes), the root-lattice census
                              root_lattices_rank4.log, the exact
                              48-point enumeration octahedral48_exact.log,
                              rigidity24.log, and the re-verification of
                              the certificate of de Laat, Leijenhorst and
                              de Muinck Keizer, llm24_certificate_check.log,
                              with its Lean file's run
                              D4InnerProducts_lean.log. The directory
                              continuation_out/ holds the best
                              configuration of every level of the third
                              pass as a 23 x 4 array (.npy). The runs used
                              numpy 2.4, scipy 1.17 (HiGHS), cvxpy 1.9,
                              Clarabel 0.11, mpmath 1.3, sympy 1.14,
                              python-flint 0.9 and Python 3.11; the endpoints of the searches
                              are local optima and a different platform
                              may reach different ones, whereas the
                              certificate is a fixed file and its
                              verification is deterministic.

  slack_continuation.py
                              Exploration of 23-point codes, not cited in
                              the paper. For
                              delta on a schedule from 0.05 down to 0,
                              estimates h(delta), the largest inradius over
                              23-point configurations with inner products
                              at most 1/2 + delta, by the optimiser of
                              inradius_search.py started from the best
                              configurations of the previous level
                              (restored to the tighter constraint) and from
                              fresh random starts. Prints one line per
                              level: h(delta), the covering radius of the
                              best configuration, the number of feasible
                              endpoints and how many are deletions, and
                              the largest inner products of the best.
                              About a quarter of an hour per pass.
                                python slack_continuation.py [fresh] [seed]

  root_meet.py
                              Supports prop:meet22 and prop:meet21 and
                              rem:meet-next, in five parts:

                                (i)   no pair of removed roots destroys a
                                      whole couple of complementary
                                      supports;
                                (ii)  512 of the 2024 triples do, in eight
                                      support patterns, four stars and four
                                      triangles;
                                (iii) the symmetry group of the root
                                      system, generated here and of order
                                      1152, is transitive on the 96 triples
                                      that are pairwise at 60 degrees;
                                (iv)  for the standard star, every
                                      direction of the doorway has first
                                      coordinate at least 1/sqrt2 and the
                                      other three nonnegative, and no two
                                      of them are 60 degrees apart unless
                                      both are removed roots;
                                (v)   one rung lower, at four removed
                                      roots, only four of the 31 orbits
                                      open a doorway at all, and in each
                                      the least achievable largest pairwise
                                      inner product among three directions
                                      kept an angle away from every root
                                      rises above 1/2 as soon as that angle
                                      is positive.

                              Parts (i) to (iii) are exact; parts (iv) and
                              (v) are sampling and minimisation, and the
                              output says so.


  multi_cap_reformulation.py
                              Supports prop:multi-cap, op:multicap
                              and rem:first-order-obstruction.
                              Checks, in order:

                                (A) Q_D is bounded for every packing-valid
                                    D of size 1 to 5, exhaustively: 24,
                                    276, 2024, 10626 and 42504 sets, no
                                    failures;
                                (B) at size 6, exactly 24 of the 134596
                                    packing-valid sets fail, and the
                                    complement of the first of them lies
                                    in a closed half-space;
                                (C) vol(V) = vol(Q_D) - vol(union of the
                                    caps that the tilted half-spaces cut
                                    from Q_D), against direct polytope
                                    volumes;
                                (D) vol(Q_D) - 8 equals |D|/3 exactly when
                                    D has no adjacent pair, and exceeds it
                                    otherwise;
                                (E) the term-by-term reduction fails: in a
                                    random search vol(E_j) comes out both
                                    above and below its single-deviation
                                    value, the shortfall exceeding 0.1
                                    against defects of order 0.3, so the
                                    single-deviation theorem gives no
                                    bound on the individual terms.

                              (A), (B) and (D) are exhaustive over the
                              stated ranges; (C) and (E) are random
                              searches and say so in the output. Eleven
                              checks, about four and a half minutes.

  polar_surface_reformulation.py
                              Supports sec:polar-surface: prop:polar-form,
                              cor:minimiser, lem:surface-form, lem:facet-inball and
                              rem:facet-local-obstruction and rem:global-routes. Eighteen checks,
                              grouped here as follows:

                                (a) the cell is the polar dual of the
                                    convex hull of the contact
                                    directions: at the root configuration
                                    the two volumes are 2 and 8;
                                (b) the cell depends on the configuration
                                    only through that hull, tested by
                                    adjoining hull points to random
                                    configurations;
                                (c) vol(V) = (1/4) * total facet 3-volume,
                                    exactly 32/4 = 8 at the roots, with
                                    all 24 facets of 3-volume 4/3;
                                (d) the same identity on configurations
                                    obtained by dropping four roots,
                                    perturbing and separating again;
                                (e) rho(g) = sqrt((1-g)/(1+g)) decreasing,
                                    rho(1/2) = 1/sqrt(3);
                                (f) every facet contains the 3-ball of
                                    radius 1/sqrt(3), by direct inradius
                                    computation;
                                (g) the facet-local bound at 24 contacts,
                                    8*pi/(3*sqrt(3)) = 4.8368, and the
                                    inscribed-ball bound pi^2/2 = 4.9348,
                                    both short of 8;
                                (h) the tight structure at a root facet,
                                    in exact rational arithmetic: eight
                                    tight neighbours, projected Gram
                                    values in {-1, -1/3, 1/3}, cutting out
                                    the regular octahedron of volume 4/3;
                                (i) enlarging the configuration never
                                    increases the cell volume;
                               (i') the square-antiprism configuration of
                                    rem:facet-local-obstruction: its nine directions in
                                    R^4 are packing-valid, its facet has
                                    3-volume 16 sqrt(2) - 64/3 =
                                    1.2940836646 against the octahedron's
                                    4/3, and the ceiling for any
                                    facet-local bound is therefore
                                    96 sqrt(2) - 128 = 7.7645, below both
                                    8 and the covering bound;
                                (j) the polar volume integral reproduces
                                    8, the Jensen bound returns 7.7351,
                                    and the Mahler-type product is 16
                                    against a conjectural 32/3.

                              (h) is exact; (j) uses Monte Carlo integration
                              over S^3 at four million samples and says so
                              in its output; the rest is double-precision
                              polytope arithmetic on exactly specified
                              configurations. The pairwise-repulsion step
                              that generates test configurations is a
                              sampling device and enters no argument.
                              Runtime under a minute.

  covering_bound.py           Supports sec:covering-bound: lem:radial-form,
                              thm:covering-bound, cor:m22,
                              prop:area-optimal and tab:covering-bound, which
                              together give thm:local-fewcontacts (the
                              local bound at any centre with at most 22
                              contacts). Nineteen checks, grouped here
                              as follows:

                                (a) the cap-area formula
                                    C(r) = pi(2r - sin 2r), against its
                                    value at r = pi and against direct
                                    sampling at four radii;
                                (b) the radial identity
                                    vol = (1/4) int sec^4(delta) at the
                                    root configuration, returning 8 and
                                    recovering the covering radius 45 deg;
                                (c) the layer-cake rewriting used in the
                                    proof, against the direct form;
                                (d) the same identity on random
                                    configurations;
                                (e) the closed form (pi m / 3) tan^3 r_m
                                    against numerical quadrature of the
                                    same estimate, agreeing to 1e-14;
                                (f) the whole of tab:covering-bound for
                                    m = 5 .. 24, and the monotonicity in m
                                    that the proof of cor:m22 uses;
                                (g) that 22 is exactly the largest m at
                                    which the bound reaches 8;
                                (h) no violation of the bound at the root
                                    configuration, at 100 of its subsets
                                    of sizes 20 to 23, or at 50 random
                                    packing-valid configurations;
                                (i) the constants quoted in the text:
                                    8.046376 at m = 22, 7.798989 at
                                    m = 24, 7.916728 at m = 23, and the
                                    implied density 0.632749;
                                (j) prop:area-optimal: the closed form for
                                    phi'(s), the convexity of phi, and
                                    the fact that equal Voronoi cell
                                    areas reproduce the global bound and
                                    minimise the per-cell sum (against
                                    1500 random area splittings);
                                (k) prop:covering-radius: every spherical
                                    Voronoi cell has circumradius at
                                    least arccos sqrt(5/8) = 37.7612
                                    degrees, checked against the
                                    configurations directly, together
                                    with the fact that it moves the
                                    volume bound only in the fifth
                                    decimal;
                                (l) the split of the shortfall at m = 24
                                    into about 0.057 of overlap and about
                                    0.140 of truncated tail, and the
                                    total overlap of about 2.0 from the
                                    96 pairs of the root system at 60
                                    degrees.

                              (b), (c) and (d) are Monte Carlo and say so
                              in the output; the table, the closed form
                              and the polytope volumes are not.
                              Runtime under a minute.

  saturation_search.py        Supports sec:m23-explicit: prop:no-transitive and
                              rem:no-transitive. Takes the covering radius as
                              the objective from the start, which is the
                              quantity the open case is about. It uses
                              that g(W) is the inradius of conv(W) about
                              the origin, so it comes off a convex hull
                              exactly. Four checks:

                                (a) the 24 roots give g = 1/sqrt(2);
                                (b) all 24 deletions give g = 1/2 exactly,
                                    with zero spread;
                                (c) no 23-point code is a single cyclic
                                    orbit: 528 rotation types, each
                                    decided by a one-variable linear
                                    programme, none feasible, smallest
                                    violation 0.0740002839 at 60 digits,
                                    attained at (a,b) = (8,17);
                                (d) direct minimisation of max_I |z_I|
                                    from perturbed deletions, random
                                    starts and partial root systems.

                              Check (c) is a proof; check (d) is
                              exploration and the output says so. Over 202
                              starts it reached 12 contact configurations,
                              every one of them a deletion. Runtime about
                              three minutes.

  covering_multiplicity.py    Supports sec:m23-explicit: prop:cov-mult
                              and prop:two-point-barrier. Two parts:

                                (a) the Cauchy-Schwarz bound on the total
                                    overlap of the 60-degree caps of a
                                    saturated configuration, 155.172054
                                    at m = 23, against 162.566121 at the
                                    deletion configuration, a slack of
                                    4.5 per cent;
                                (b) a linear programme over all measures
                                    on [60, 180] degrees of mass 253
                                    subject to Bonferroni at 24 radii and
                                    Gegenbauer positivity at 12 degrees,
                                    whose minimum of the pair weight is
                                    0.041573648 against the 0.083272432
                                    that prop:pair-budget needs.

                              Part (b) is the sharp statement about the
                              route: nothing reading only the pair angles
                              gets past half way. Runtime about four
                              minutes.

  pair_budget.py              Supports sec:m23-explicit: prop:pair-budget
                              and rem:pair-budget. prop:second-order evaluates
                              the pairwise estimate at the deletion
                              configuration and gets 7.997885, three
                              pairs short of 8. The deletion
                              configuration is extendable, so it is not
                              one of the configurations Problem 7.33 is
                              about, and this script computes what the
                              same estimate would need from one that is.
                              Five checks:

                                (a) the per-pair weight w(gamma) and its
                                    collapse away from 60 degrees;
                                (b) the 7.997885 of the deletion
                                    configuration recovered from its 88
                                    tight pairs;
                                (c) that 91 pairs at 60 degrees carry the
                                    estimate to 8.000652 while 90 give
                                    7.999729;
                                (d) the elementary degree bound of 10 on
                                    S^2, hence at most 115 tight pairs on
                                    23 directions;
                                (e) that 91 lies strictly between 88 and
                                    115.

                              The quadratures are adaptive; nothing here
                              is Monte Carlo. Runtime about two minutes.

  spherical_code_23.py        Exploration, not cited in the paper; the
                              codes inside the 600-cell are settled exactly
                              in sec:m23-codes. Contact
                              configurations of m directions are
                              spherical codes of m
                              points on S^3 of minimal angle at least 60
                              degrees, and the script measures how much
                              room such a code has. Riesz continuation,
                              minimising sum |w_i - w_j|^{-s} over the
                              sphere with s = 4, 16, 64, 256, 1024,
                              followed by a direct softmax reduction of
                              the largest inner product. Four sizes:

                                m = 24, where the answer is 1/2;
                                m = 22, where a code with room to spare
                                    exists;
                                m = 25, where no contact configuration
                                    exists;
                                m = 23, from 1500 random starts.

                              At m = 23 it reports the smallest largest
                              inner product reached, the number of
                              starts that fell below 1/2, and, for every
                              outcome within 5e-3 of feasibility, the
                              multiset of Gram entries, which is an
                              O(4) invariant and therefore separates
                              configurations not related by an isometry.
                              Floating point throughout. This is
                              exploration and the output says so; it is
                              not a proof that no saturated 23-point
                              configuration exists. Runtime about forty
                              minutes.

  extendability.py            Supports sec:m24: thm:m24,
                              prop:extendable and cor:remaining. Six
                              checks:

                                (a) the root configuration has
                                    circumradius sqrt(2), g = 1/sqrt(2),
                                    cell volume 8, and is saturated, as
                                    a 24-point kissing configuration must
                                    be;
                                (b) the root system minus one root has
                                    circumradius exactly 2, g exactly
                                    1/2 and cell volume 25/3, and the
                                    deleted root is what extends it, so
                                    it sits on the boundary of the
                                    criterion;
                                (c) the three forms of the criterion
                                    (admissible extra direction,
                                    circumradius at least 2, covering
                                    radius at least 60 degrees) agree on
                                    random configurations;
                                (d) the root configuration has Gram
                                    values in {-1,-1/2,0,1/2}, which is
                                    what Lemma 5.1 of de Laat,
                                    Leijenhorst and de Muinck Keizer
                                    asserts of every 24-point
                                    configuration;
                                (e) the numerical chain of cor:remaining;
                                (f) a search for a saturated 23-point
                                    configuration, over the 24 one-root
                                    deletions and random starts, which
                                    found none: the largest g attained
                                    was 1/2 itself, and the random
                                    starts reached no 23-point contact
                                    configuration at all. This is
                                    exploration, labelled as such in the
                                    output, and is not evidence that none
                                    exists.

                              Runtime about 3 minutes.

  second_order_estimate.py    Supports lem:no-triples and prop:second-order:
                              the covering estimate with every pairwise
                              overlap put back. Four checks, the
                              quadratures in 30-digit arithmetic:

                                (a) three contact directions have
                                    circumradius at least
                                    arccos sqrt(2/3) = 35.2644 degrees,
                                    by the Rayleigh-quotient argument and
                                    against 40000 random admissible
                                    triples;
                                (b) r_23 = 34.6106 and r_24 = 34.0987 both
                                    lie below that, so below r_m the caps
                                    meet only in pairs and
                                    inclusion-exclusion is exact;
                                (c) the closed form for the lens measure
                                    against direct sampling on S^3;
                                (d) the resulting bound: 7.997885 at
                                    m = 23 and 7.858738 at m = 24, both
                                    below 8. At m = 23 only 0.002115 of
                                    the target is left unaccounted for,
                                    and no argument built from cap
                                    measures and pairwise intersections
                                    can recover it.

                              Runtime a few seconds.

  local_cell_obstruction.py   Supports rem:no-local-cell: the shape deficit
                              that prop:area-optimal leaves cannot be
                              collected one cell at a time. Five checks:

                                (a) at the root configuration the mean of
                                    sec^4 over a spherical Voronoi cell
                                    is 16/pi^2 = 1.6211389, so the
                                    per-cell inequality is sharp;
                                (b) nine directions of R^3 with pairwise
                                    inner products at most 1/3 exist, so
                                    nine contacts at 60 degrees are
                                    admissible; the cell of such a
                                    direction has mean 1.5732, and 1.5954
                                    when the nine are moved out to 61
                                    degrees, where every inner product is
                                    strictly below 1/2;
                                (c) on a 22-point configuration there are
                                    directions whose nearest contact lies
                                    outside the Delaunay cell containing
                                    them, missing every vertex of that
                                    cell by more than 0.1 in cosine: the
                                    Delaunay decomposition does not
                                    localise the integrand at all;
                                (d) the closed form
                                    4 E[(1+M)^-4] / E[(1+q)^-2], with
                                    E[(1+M)^-4] = 1/5 exactly, gives
                                    1.543643 at the regular simplex of
                                    edge 60 degrees, 4.78 per cent short
                                    of the target, against direct
                                    sampling at edge 60 and 62 degrees;
                                (e) the Delaunay cells of D4 are 24
                                    congruent spherical octahedra of
                                    circumradius 45 degrees, and no four
                                    roots are pairwise at 60 degrees, so
                                    the regular simplex occurs nowhere in
                                    D4 though it does occur as a Delaunay
                                    cell of a contact configuration.

                              The integrals over S^3 are Monte Carlo and
                              the output says so; the margins are
                              percentages, not last digits.
                              Runtime about 30 seconds.

  rigidity23.py               Supports prop:deletion-rigid, the
                              infinitesimal rigidity of the deletion
                              configuration. Five checks, all in exact
                              integer arithmetic on the unnormalised
                              roots, with no floating-point step:

                                (a) deleting one root leaves 23
                                    directions and 88 tight pairs, the
                                    four values of <a_i,a_0> occurring
                                    1, 8, 6 and 8 times;
                               (a') the six pair types listed in the
                                    proof are the only ones that occur
                                    among the 88;
                                (b) the stress weights y_ij, which are
                                    1, 2 or 3 according to the pair
                                    (<a_i,a_0>, <a_j,a_0>), are strictly
                                    positive on every tight pair;
                                (c) they satisfy the equilibrium relation
                                    sum_j y_ij a_j + mu_i a_i = 0 for
                                    every i, the 23 x 4 residual matrix
                                    being exactly zero;
                                (d) the space of motions holding all 88
                                    pairs at equality has dimension
                                    exactly 6, and the infinitesimal
                                    rotations already span it.

                              Together (b), (c) and (d) say that every
                              first-order motion of the deletion
                              configuration through contact
                              configurations is a rotation.
                              Runtime well under a second.

  rigidity_spectrum.py        Supports lem:rigidity-spectrum and thm:local-uniqueness.  In
                              integer arithmetic: 4N annihilates
                              x(x-8)(x-20)(x-24)(x-32) and the ranks of 4N - cI
                              (the spectrum of the rigidity operator); the
                              projection onto the image of the operator has
                              every diagonal entry 11/16; and a single displaced
                              direction has ratio sqrt(96/11).  Under a second.
                              Log: runs/rigidity_spectrum.log.

  rigidity24.py               Supports the corollary after prop:deletion-rigid, the same statement for the root
                              system itself with nothing deleted. Four
                              checks in exact integer arithmetic: 24
                              directions and 96 tight pairs, every
                              direction in 8 of them; the constant
                              stress (weight 1 on every tight pair, -4
                              on the diagonal) is positive on the pairs;
                              it is in equilibrium, the eight
                              neighbours of every root summing to four
                              times the root; and the space of motions
                              holding all 96 pairs at equality has
                              dimension exactly 6, spanned by the
                              rotations. Runtime a few seconds.

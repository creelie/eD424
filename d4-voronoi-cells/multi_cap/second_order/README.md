# The second order at the root system

The scripts of sec:second-order of the paper (lem:second-variation to prop:push-integrated, fig:second-order).
Run them from this directory.

| file | what it does |
| --- | --- |
| `rational_model.py` | the form H of the second variation, the 120 rows of the first-order packing cone and c = 2 1_eta, in exact rationals from the integral root system |
| `verify_cone_certificate.py` | the proof of m = -1: H + c c^T = P + B^T N B with N >= 0 and P positive semidefinite, by an exact LDL^T (three seconds, no floating point) |
| `exact_certificate.pkl`, `.txt` | the certificate: N constant on 43 orbits of the signed-permutation group, 35 values positive |
| `cone_min.py` | m in floating point: the best feasible point over 62 starts, and the Shor relaxation with the products of the cone rows, both -1 |
| `certify_m.py`, `exact_cert.py`, `exact_cert2.py`, `exact_cert3.py` | how the certificate was found and made exact |
| `hessian.py` | the form in floating point (writes `H.npy`), its spectrum, and a finite-difference check |
| `exact_push.py` | the one-centre formula, lem:pure-push and prop:push-integrated on examples |
| `untilt.py` | tilting is not monotone: 103 of 237 tilted packings below their untilted versions, by up to 3.6e-5 |
| `remainder_test.py` | 300 packings with tilts on the edges of the cone: vol - 8 - (2/3) S + (1/2) S^2 >= 0 on all; writes `runs/remainder_rows.dat` |
| `tilt_block.py` | the 72 x 72 tilt block of H (the Hessian of the contact-cell volume in the directions alone) is positive semidefinite, by an exact LDL^T with 15 zero pivots; its kernel is the 6 rotations and 9 further directions, and the infinitesimal strains are not in it; writes `tilt_kernel.npy` |
| `tilt_quartic.py` | the contact-cell volume as a function of the tilts, the rotation-free coordinates (the 9 flat directions and the 57 others) and the relaxed objective (vol - 8)/t^4 (floating point); used by `contact_valley.py` |
| `hexagon_loop.py` | prop:hexagon-loop: turning one A2 hexagon of roots by theta in its plane and tilting the other 18 to sin psi = 4C/(4C^2+3), C = cos(pi/6 - theta), keeps the contact-cell volume exactly 8 on a closed curve through the root system; exact (sympy: the Gram matrix at theta = 0, the integrals J1, J2 in closed form, the perfect square, the slice conditions, the largest inner product), with qhull at 13 points (9e-15), 200 random points of the two-parameter family, and 50-digit volumes at 3 points (1e-49); the algebra again in `lean/D4HexagonLoop.lean` |
| `contact_valley.py` | rem:contact-valley: along straight lines in the 9 flat directions the volume grows at fourth order, but with the other tilts relaxed the minimiser follows a curve along which vol - 8 stays below 3e-10 up to tilt 0.45 and the Hessian has one zero eigenvalue beyond the rotations: the curve of prop:hexagon-loop; floating point, exploration |
| `contact_cell_scan.py` | the least contact-cell volume over 24 unit directions with no packing constraint: 7.96553 < 8, at directions with inner products up to 0.579 (24 facets, 99 vertices; saved in `contact_cell_min.npy`), so a bound vol(Q_w) >= 8 can only hold near the root system or under the packing constraint; floating point, exploration |
| `contact_cell_constrained.py` | the least contact-cell volume over 24 unit directions with inner products at most 1/2 + s, exact gradient -A_i c_i: 8 (the root system and the curves of prop:hexagon-loop) for s = 0.005 and 0.01; codes away from the root system with larger cells (least 8.0679) at s = 0.02; cells of volume 7.99802 and 7.99288 at s = 0.03 and 0.035; floating point, exploration |
| `code24_second.py` | the best 24-point code other than the root system: from 62 codes of slack 0.03, 59 minimisations of the largest inner product stop at 0.516978 (slack 0.016978), a code with no antipodal pair, 84 pairs at that value, five kinds of point, contact cell 8.0948; saved in `code24_second.npy`; floating point |
| `code24_exact.py` | prop:second-code, exact: that code moved to rational points of S^3 (inverse stereographic projection), every inner product at most 1/2 + 849/50000, one at distance at least 57/250 from -1, -1/2, 0, 1/2, 1, so d(W) >= 57/500; writes `code24_exact.txt` for `lean/D4SecondCode.lean` |
| `cell_hull_search.py` | rem:what-g-needs: the statement (G), vol(V(Y) ∩ K(Y)) >= 8 for 24 centres within sqrt6 with pair-term bound T(Y) <= 8. Without the condition T <= 8 a packing reaches 4.750 (T = 10.82; `cell_hull_free_example.npy`); with it, no random configuration enters the region, and the 20 of 64 starts inside it that stay there all end at the root system with volume 8; floating point, exploration |
| `hole24.py` | the deepest hole of a 24-point code of slack s: the least max_i <theta, w_i> falls linearly, 0.70711 at s = 0 (the root system), 0.69280, 0.67817, 0.66322, 0.64409, 0.63238, 0.58327 at s = 0.004, 0.008, 0.012, 0.017, 0.02, 0.03. A further centre within sqrt6 beside 24 centres within 2.0161 needs 0.6141, so numerically the room for a 25th centre opens only near slack 0.025; floating point, exploration |
| `independent_check.py` | recomputes every volume by intersecting halfspaces: H against second differences (6e-8), the cone rows against exact distances (2e-9), the spectrum, the one-centre formula (1e-14), lem:pure-push on 400 push patterns |

Logs are in `runs/`.  What is proved here is the second order (the certificate)
and the push-out statements (lem:pure-push, prop:push-integrated, by the proofs in the
paper); the remainder and the non-monotonicity are floating-point experiments.

Two statements of the draft these files came with are corrected in the paper:
the eigenvalue -6.194 of H has multiplicity 8, not 6, and on the push-outs H has
matrix Adj - 4I, not 2 Adj - 4I (its form is 2 sum over edges eta_i eta_j - 4 sum eta_i^2).

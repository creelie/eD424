# cap_certificate/

The cap inequality of sec:cap-theorem, end to end.

## The cap inequality in one script

  cap_inequality_certificate.py
                              Reproduces the whole of sec:cap-theorem end to
                              end, in the order the section proves it:

                                (A) exact rational vertex enumeration of
                                    Q, giving 25 vertices and vol(Q)=25/3,
                                    with the pyramid of volume 1/3
                                    recovered as the difference from the
                                    24-cell;
                                (B) the exact check that every vertex of Q
                                    has l^1 norm at most sqrt(2) in the
                                    orthonormal root frame, which is the
                                    enclosure Q inside B, together with the
                                    fact that the bound is attained;
                                (C) the closed form for the cap of the
                                    cross-polytope as a divided difference
                                    of a(2 sqrt a - 1)_+^4, checked against
                                    directly computed polytope volumes at
                                    60 random directions;
                                (D) the symbolic identity for the fourth
                                    derivative, 3(20t^2-3)/(2t^5) at
                                    a = t^2, and the vanishing of the third
                                    derivative at a = 1/4, which together
                                    give the monotonicity of the divided
                                    difference in each node;
                                (E) exact real-root isolation for the three
                                    polynomial inequalities covering the
                                    case of at most one node above 1/4;
                                (F) the hand proof of the case of two or
                                    more nodes above 1/4: the fifth
                                    derivative of g (so the third
                                    derivative is concave beyond 1/4 and
                                    below its tangent 96(a - 1/4)), the
                                    five vertices of the region by exact
                                    enumeration, and the exact values 2/9,
                                    3/16, 1/8, 27/256, 0 of h[a] there, so
                                    F <= 8/9; a sampled floating-point
                                    cross-check of F <= 4 h[a];
                                (F') an independent exact check of the same
                                    case by adaptive subdivision: 303
                                    boxes, corner bounds entirely in
                                    fractions.Fraction, largest value
                                    0.99755050;
                                (G) an independent check of the conclusion:
                                    directly enumerated Voronoi cell
                                    volumes at 300 random (tilt, direction)
                                    pairs, all with nonnegative defect.

                              Steps (A), (B), (D), (E), (F) and (F') are
                              exact, apart from the cross-check at the end
                              of (F); (C) and (G) are double-precision
                              cross-checks that confirm the exact work and
                              are not relied on by it. The script prints
                              PASS or FAIL for each of its 27 checks and
                              exits nonzero if any fails. Runtime about
                              ten seconds.

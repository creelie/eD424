# analysis: two routes to (G) and (C), and where each stops

Statements (G) and (C) are set out in `../d4-voronoi-cells/README.md`.
Together they would prove that every Voronoi cell of a unit-ball packing in
R^4 has volume at least 8, and so that D4 is the densest packing in four
dimensions. This directory records two attempts at an
argument that uses functions and inequalities instead of case enumeration,
with the computations that show where each one stops. Neither closes (G) or
(C).

## (G): rearrangement of the facet regions

`cap_rearrangement.py`. Write the cell in polar coordinates and give each
facet the region of directions whose ray leaves the cell through that facet.
For a region of given area, the cone volume is least when the region is a
round cap about the facet normal, and that least value is convex in the area.
So every cell with N facets at distance at least 1 has

    vol(V) >= N F(1, 2 pi^2 / N),   F(1, A) = (pi/3) tan^3 psi,
    2 pi (psi - sin psi cos psi) = A.

This is a proof, and at N <= 22 the bound exceeds 8. At N = 24 it gives
7.798989, short of 8 by 0.201. The 24-cell itself has octahedral regions
whose cones have volume 1/3, against 0.324958 for the round cap of the same
area. A bound of this kind cannot be made sharp at 8: it would have to be
exact at the root system, and so would have to see the octahedral shape of
the regions. `contact_cell_scan.py` in `d4-voronoi-cells/multi_cap/second_order`
shows the same thing from the other side: without the packing condition, 24
facets at distance 1 enclose as little as 7.96553.

## (C): functions of one inner product

`delsarte_slack.py`. If (C) failed at 25 centres, the certificates of
`prop:C-radial` would put at least 23 of them within 2.2 of the centre, 22
within 2.15, 21 within 2.1 and 17 within 2.05. Two centres within rho of the
centre and 2 apart have directions with inner product at most 1 - 2/rho^2. A
contradiction drawn from a function of one inner product would have to show
that such codes have fewer than 25 points. The best such function, Delsarte's
linear programme, gives

| rho | 2 | 2.0161 | 2.05 | 2.1 | 2.15 | 2.2 |
| --- | --- | --- | --- | --- | --- | --- |
| largest inner product | 0.5000 | 0.5080 | 0.5241 | 0.5465 | 0.5673 | 0.5868 |
| Delsarte bound | 25.56 | 26.20 | 27.66 | 30.15 | 32.73 | 35.15 |

Musin's refinement, a function that may be positive near -1 with the points
there counted by hand, lowered the bound at 2 from 25.56 to below 24.865
(Ann. of Math. 168 (2008), 1-32,
[doi:10.4007/annals.2008.168.1](https://doi.org/10.4007/annals.2008.168.1)).
A gain of that size leaves every other column above 25. The three-point
programmes of `d4-voronoi-cells/gap_closure` see triples and the radii, and
their optima on sampled constraints at 25, 26 and 27 centres also lie above
the level that (C) needs. So the counts alone do not decide (C). What
decides it is how the close centres sit: near a root system, with the
far centres in its holes.

## What a proof needs

Both routes lead to the same missing statement: a stability theorem saying
that 21 to 24 centres within about 2.2 of a centre, pairwise 2 apart, lie
close to part of a root system D4, with an explicit tolerance. The code in
`d4-voronoi-cells/multi_cap/stability` proves such a statement only for
centres within 2.0161. Between 2.0161 and 2.2 no tolerance is known. The
only method known to see the root system exactly is the second level of the
Lasserre hierarchy, which `d4-voronoi-cells/level2` runs at distance exactly
2. Running it with slack, or finding a geometric stability argument, is the
open step.

All values here are floating point except the rearrangement bound itself,
whose proof is the argument above. The logs are in `runs/`.

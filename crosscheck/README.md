# crosscheck: the two packages against each other

`two_translates.py` runs the code of `d4-voronoi-cells/` on the packings of
the paper in `paper/`, whose centres form a lattice and one translate of it,
and checks the two against each other. Run it from the repository root:

    python3 crosscheck/two_translates.py [random] [near] [seed]

`runs/two_translates.log` is the output of `python3
crosscheck/two_translates.py 300 300 1` (under a minute). On 943
packings (the 22 candidates of Proposition 5.4, 300 random two-periodic
packings, 900 perturbations of sqrt2 D4, less 33 whose cells were too long to
enumerate):

- the Voronoi cell volume from the half-spaces agrees with sqrt(det Q)/2
  from the Gram matrix to 3e-15, and is at least 8, with 8 only at sqrt2 D4;
- `prop:inversion-hull`, vol(V_0) >= vol(V(Y) n K(Y)), holds at all 917
  packings where K(Y) has interior;
- statement (G) holds at the 54 packings with 24 centres within sqrt6 and
  T <= 8, with least value 8 at sqrt2 D4;
- none has 25 or more centres within sqrt6. Forced to have 25, from 48
  starts near sqrt2 D4, the local minima of T all lie above 8; the least,
  8.658237, has 26 centres within sqrt6, 18 of them at distance 2, and
  det Q = 336. So (C) holds on every two-periodic packing found.

All of it is floating point.

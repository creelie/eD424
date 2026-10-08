# Two gaps in the reduction from a Voronoi cell to a contact configuration

*Written against v1.3.0, whose numbering it uses.  v1.4.0 corrects both
lemmas, proves the bound for the Voronoi cell of a packing whenever the
centres within 2 sqrt 2 all touch it or pass the distance criterion
(Section 2.7 of the paper, `multi_cap/shell_reduction.py`), and states the
remaining case, a centre crowded by near-contacts, as Conjecture 1.6;
v1.5.0 narrows that case further (Section 2.8) and proves the bound near
the contact regime, and v1.6.0 proves it for every centre with at most
twenty-three centres within sqrt 6.  See the addenda of `../REPORT.md`.*

Everything the paper proves about volumes after Section 2 is stated for
*contact configurations*: finite sets W of unit vectors with pairwise inner
products at most 1/2 (Definition 7.7, Proposition 7.8 "Polar form",
Corollary 7.17, Theorem 7.25, Corollary 7.39, Theorem 7.40).  The main theorems, Theorem 1.5 "General local bound"
(`thm:local-general`) and Theorem 1.6 "General density bound"
(`thm:main-general`), are about the Voronoi cell of a centre in an arbitrary
unit-ball packing.  The bridge between the two is Layers 1 and 3 of the
introduction: Lemma 2.7 "Shell localisation" (`lem:shell-new`, Section 2.4) and
Lemma 2.9 "Monotonicity in the contact radii" (`lem:radial-new`,
Section 2.6).
Both lemmas, as stated and used, are false in general, and nothing else in the
paper closes the gap they leave.  This does not show the theorems are false.
In every example below the true cell has volume at least 8.  What it shows is
that the paper does not prove them for all packings.

## Gap 1: the all-contact corner is not a contact configuration

`lem:radial-new` moves every active neighbour (every centre closer than
2 sqrt 2, Notation 2.6) radially in to distance 2 and asserts that "among all
packing-valid radius assignments for a fixed active-direction pattern,
vol(V_c) attains its minimum at the all-contact corner".  Monotonicity is
correct: pulling a hyperplane in only shrinks the cell, so the corner cell is a
lower bound.  But the corner is packing-valid only if the active directions are
pairwise at least 60 degrees apart.  Neighbours at distances d1, d2 > 2 need
only |y1 - y2| >= 2, which allows directions as close as
arccos((d1^2 + d2^2 - 4) / (2 d1 d2)): 49.2 degrees at d = 2.4, and 41.4
degrees as d approaches 2 sqrt 2.  The paper then treats the corner as a
contact configuration.  It says so in the proof of Proposition 7.8 ("a
configuration at the all-contact corner with a full active set is a contact
configuration") and in Corollary 7.31 (`cor:conj-resolved`).  Theorem 1.3
(`thm:local-fewcontacts`) likewise assumes a packing already at the corner and
defers to `lem:radial-new` for why "this is the case that matters".

* `layer3_minimal.py`: two neighbours at distance 2.4, exactly 2 apart, at
  49.25 degrees (inner product 0.653).  The configuration is a valid packing.
  At the corner the two are 1.667 apart, which is not a packing and not a
  contact configuration.  Distance 2.4 is below 2R for both truncation radii
  of Proposition 7.68, R = sqrt(3/2) (2R = 2.449) and R = sqrt(8/5)
  (2R = 2.530), so these neighbours cut the ball in which the truncated
  estimates of Sections 7.4 and 7.11 are taken.
* `layer3_example.py`: 48 neighbours at distance 2.6131 in the 48 directions of
  the binary octahedral group, a valid packing of 49 balls (least distance
  exactly 2).  At the corner the 48 directions have inner products up to 0.707
  and 48 > 24 elements.  The corner cell has volume **6.594 < 8**.  The true
  cell has volume 19.216.

So the chain "true cell >= corner cell >= 8" breaks at the second inequality.
The corner cell can be far below 8, and the paper's results about contact
configurations do not apply to it.  A proof has to bound the true cell of a
packing whose active neighbours are not all at contact distance.  Every
neighbour in the shell (2, 2R) cuts the truncation ball with a smaller cap,
but there can be more than 24 such neighbours and they can sit closer than 60
degrees.  That is a different extremal problem from the one the paper solves.

## Gap 2: neighbours at distance >= 2 sqrt 2 can cut the cell

`lem:shell-new` states that "only neighbours y of a centre c with
|y - c| in [2, 2 sqrt 2) can affect vol(V_c)".  The reason given is that such
a half-space contains the reference 24-cell.  That shows the far neighbour
cannot cut *the 24-cell*.  It does not show it cannot cut V_c, which is larger
than the 24-cell whenever V_c is not the reference cell itself.

* `shell_example.py`: the 23-root deletion configuration (cell volume 25/3,
  circumradius 2) together with one neighbour at distance 2.9 > 2 sqrt 2 in
  the direction of the deleted root.  The 25 centres form a valid packing.
  The far neighbour cuts the cell from 25/3 to 1328453/160000 = 8.30283, the
  exact value 25/3 - (1/3)(2 - 1.45)^4, agreed by the polytope computation.

For the 22-, 23- and 24-contact arguments this matters less.  They bound the
cell only inside a ball of radius at most sqrt 2, which far neighbours cannot
reach.  It matters wherever the whole cell is used.  That includes the
single-deviation theorem (Theorem 1.1, `thm:local-uncond`), whose hypothesis
constrains only the active neighbours.

* `single_deviation_probe.py`: a numerical search over configurations of 23
  roots at distance 2, one deviated neighbour at distance >= 2, and one free
  neighbour at distance >= 2 sqrt 2, all pairwise >= 2 apart, minimising the
  true cell volume.  Over at least 131 random starts the least volume found
  was 8.000000000, at the D4 configuration (`single_deviation_probe.log`).
  This is evidence that Theorem 1.1 survives the gap.  It is not a proof.

## What would close them

Gap 2 is probably repairable in the multi-contact arguments by recording that
the truncation radius is at most sqrt 2.  In the single-deviation theorem it
needs an argument bounding what a far neighbour can remove from the part of
the cell outside the ball of radius sqrt 2.  Gap 1 needs the volume bound for
cells whose active neighbours lie at distances in (2, 2 sqrt 2) with
directions closer than 60 degrees.  That is not a technicality: it is where
non-contact neighbours enter the 24-cell conjecture, and the paper, as
written, does not treat it.

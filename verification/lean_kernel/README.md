# Proposition 5.1 checked in the Lean kernel

This directory holds a Lean 4 proof, with Mathlib, of the finite part of the paper: the ten
vertices of R, all 1137 edges at them, and the behaviour of det J and det Q along every edge.
Every step is checked by the Lean kernel. There is no `native_decide`, no `sorry` and no new
axiom; the main theorem depends only on `propext`, `Classical.choice` and `Quot.sound`.

## The statement

`D4.certificate` in `lean/D4Lean/Main.lean`. The objects are defined in
`lean/D4Lean/Defs.lean` over the real numbers of Mathlib: `Mstar` is
M* = (Z⁴ × {0} \ {0}) ∪ (Z⁴ × {±1}), `qform J k` is J[k], `Rset` is R, `Qb J` is the leading
4 × 4 block and `Gamma` is Γ. For each of the ten vertices `vtx i` and its list of edges
`edgesOf i` the theorem gives:

* `VFacts`: J ∈ R; the k ∈ M* with J[k] = 4 are exactly the listed minimal vectors up to
  sign; det J and det Q are the listed values; det J > 0 or det Q ≥ 256; and det J = 0,
  det Q = 256 exactly at the marked vertex J_0.
* `CFacts`: J is a vertex of R (a symmetric X with X[k] = 0 at every minimal vector is 0),
  and every symmetric X with X[k] ≥ 0 at every minimal vector is a nonnegative combination
  of the directions Y of the listed edges. So the listed edges are all the edges of R at J.
* `EFacts`, for every listed edge J + tY: along the edge (0 ≤ t ≤ t* for a bounded edge,
  t ≥ 0 for an unbounded one) det(J + tY) > 0 or det Q(J + tY) ≥ 256, and
  det = 0, det Q = 256 only at the listed points. A bounded edge ends at
  J + t*Y = Tᵀ J_j T with T ∈ Γ and j one of the ten vertices, and leaves R after t*; an
  unbounded edge has Y[k] ≥ 0 for every k ∈ M*, so it stays in R.

Together with the paper (Corollary 2.11 and Proposition 3.2) this is the computation behind
Propositions 5.1 and 5.4. The reduction to vertices and edges, the closure argument and the
equality case are proved by hand in the paper and are not part of the Lean proof.

The ten vertices are, up to Γ, the matrices J_0, ..., J_9 of Table 5.1;
`compute/table.py` prints the matching (vertices 4, 5, 6 here are J_6, J_4, J_5).

## How the proof is organised

* `lean/D4Lean/Check.lean` (core Lean only): integer checkers that return `Bool`.
* `lean/D4Lean/Sound/*.lean`: what a `true` answer means, proved over ℝ with Mathlib.
  `Cone.lean` proves that a pointed cone {x : aᵢ · x ≥ 0} is spanned by vectors y whenever
  every set of 14 rows with a nonzero common kernel vector is certified, by a vector that
  takes both signs on the rows or by one of the y up to sign.
* `lean/D4Lean/Cert/*.lean`, `lean/D4Lean/Data/*.lean`: the certificate, written by
  `lean/generate/make_lean.py` from `cert/certificate.txt`; each piece is a theorem
  `... = true` proved by `decide +kernel`, that is by evaluation in the kernel.

The certificate data: for each vertex a box outside which J[k] > 4, certified by a lower
bound for J; fifteen minimal vectors whose equations determine J; an elimination
certificate for det J and det Q. For each edge: the end point and T, or the rank-one form
of Y; det J(t) and det Q(t) as polynomials, by interpolation at six points with
elimination certificates; break points, and on each piece between them a coefficient test
showing det J > 0 or det Q > 256. For the cone at each vertex: a rank certificate, the
hyperplanes spanned by 14 rows, and for every set of 14 rows a hyperplane containing it.

## Files

| Path | Contents |
| --- | --- |
| `compute/enumerate.py` | the vertices of R up to Γ and the edges at them (cddlib, exact) |
| `compute/equality.py` | the points with det J = 0 and det Q = 256 |
| `compute/table.py` | the matching with Table 5.1 |
| `data/` | the output of `enumerate.py` and `equality.py` |
| `cert/make_cert.py`, `cert/certificate.txt` | the certificate as a text file |
| `lean/` | the Lean project (Lean 4.34.1, Mathlib 4.34.1) |
| `check.sh` | regenerates everything, compares it with the committed files and builds |

## Run

```sh
verification/lean_kernel/check.sh          # regenerate, compare, lake build, print the axioms
verification/lean_kernel/check.sh --full   # also recompute data/ from scratch
```

`check.sh` needs Python 3 with sympy, numpy and pycddlib, and elan. The build checks
1137 edges and ten vertex cones. On four cores `check.sh --full` takes 8 minutes (27 minutes
of processor time) after Mathlib is downloaded; one Lean process needs at most 6.6 GB of
memory, and the parallel build at most 11.6 GB.

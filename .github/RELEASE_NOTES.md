**Densest packings of two translates of a lattice in four dimensions**
Deep Bhattacharjee

Let Λ ⊂ R⁴ be a lattice and b ∉ Λ such that distinct points of
Λ ∪ (Λ + b) are at distance at least 2. Then covol(Λ) ≥ 16, with equality
only when Λ ∪ (Λ + b) is congruent to √2 D₄. So no packing of unit balls
whose centres form two translates of a lattice is denser than D₄, and every
Voronoi cell of such a packing has volume at least 8, the volume of the
regular 24-cell.

The general 24-cell conjecture and the optimality of D₄ among all packings
in R⁴ remain open and are not claimed.

### Files

* `densest-two-translates-d4.pdf`: the paper (12 pages, amsart)
* `densest-two-translates-d4-tex.zip`: LaTeX source with the figures as PNG
* `densest-two-translates-d4-arxiv.tar.gz`: LaTeX source with the figures as PDF

### Verification

The one computational statement, the vertex and edge table of the polyhedron
of two-periodic packings (Proposition 5.1), is exact and re-checked by four
independent programs: C (integers, including all extreme rays from
14-subsets of minimal vectors), Julia (rationals, Sturm sequences), Lean 4
(kernel proof of the sign lemma and evaluation of every certificate) and
Python (sympy). Run `verification/shell/run_all.sh`.

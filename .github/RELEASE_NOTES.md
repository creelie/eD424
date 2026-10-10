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

### New in v1.1.1

* The run logs of `crosscheck/` and `analysis/` are in the archive. Their
  READMEs point to `runs/`, but a `.gitignore` rule had kept the logs out.
* In the paper, the agreement of the ten vertices with the numerical
  enumeration of Andreanov and Kallus is a footnote. Nothing else in the
  paper changes.

v1.1.0 added the manuscript in the form submitted to Discrete & Computational
Geometry, a copy of the d4-voronoi-cells v2.0.0 code
([doi:10.5281/zenodo.23240354](https://doi.org/10.5281/zenodo.23240354)), the
floating-point cross-check of (G) and (C) on 943 two-periodic packings, and
two analytic approaches to (G) and (C) with the points where each stops.

### Files

* `densest-two-translates-d4.pdf`: the paper (13 pages, amsart)
* `densest-two-translates-d4-tex.zip`: LaTeX source with the figures as PNG
* `densest-two-translates-d4-arxiv.tar.gz`: LaTeX source with the figures as PDF

### Verification

The one computational statement, the vertex and edge table of the polyhedron
of two-periodic packings (Proposition 5.1), is exact and re-checked by four
independent programs: C (integers, including all extreme rays from
14-subsets of minimal vectors), Julia (rationals, Sturm sequences), Lean 4
(kernel proof of the sign lemma and evaluation of every certificate) and
Python (sympy). Run `verification/shell/run_all.sh`.

### Citation

Concept DOI (all versions):
[10.5281/zenodo.23239767](https://doi.org/10.5281/zenodo.23239767).
Version 1.1.1: [10.5281/zenodo.23288720](https://doi.org/10.5281/zenodo.23288720).
Version 1.1.0: [10.5281/zenodo.23242482](https://doi.org/10.5281/zenodo.23242482).

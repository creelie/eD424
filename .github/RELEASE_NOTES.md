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

### New in v1.1.0

* The manuscript in the form submitted to Discrete & Computational Geometry:
  a 169-word abstract, a Statements and Declarations section, the AI-use
  statement in Appendix A, the Zenodo archive cited as data, and full DOI
  links. The mathematics is unchanged. `scripts/build_submission.sh` writes
  the PDF and a source zip with the figures as Fig1 and Fig2 (PDF and EPS).
* `d4-voronoi-cells/`: a copy of the d4-voronoi-cells v2.0.0 code
  ([doi:10.5281/zenodo.23240354](https://doi.org/10.5281/zenodo.23240354)),
  under its own `LICENSE`, without its manuscript.
* `crosscheck/`: the statements (G) and (C) of that code tested on 943
  two-periodic packings. Cell volumes from the Gram matrix and from the
  half-spaces agree to 3e-15, (G) holds wherever it applies, and packings
  forced to have 25 centres within √6 keep T above 8. Floating point.
* `analysis/`: two analytic approaches to (G) and (C) and where each stops.
  Rearranging the facet regions proves only vol ≥ 7.798989 for a cell with 24
  facets, and the Delsarte bound for the codes (C) needs is 27.7 to 35.2
  points. Both lead to the same open step, a stability theorem for 21 to 24
  centres within about 2.2.
* The CI jobs now time out after 30 minutes.

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

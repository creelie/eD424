**Densest packings of two translates of a lattice in four dimensions**
Deep Bhattacharjee (corresponding author) and Ushashi Bhattacharya

Let Λ ⊂ R⁴ be a lattice and b ∉ Λ such that distinct points of
Λ ∪ (Λ + b) are at distance at least 2. Then covol(Λ) ≥ 16, with equality
only when Λ ∪ (Λ + b) is congruent to √2 D₄. So no packing of unit balls
whose centres form two translates of a lattice is denser than D₄, and every
Voronoi cell of such a packing has volume at least 8, the volume of the
regular 24-cell.

The general 24-cell conjecture and the optimality of D₄ among all packings
in R⁴ remain open and are not claimed.

### New in v2.0.0

* Ushashi Bhattacharya (formerly National Taiwan University) is the second
  author; Deep Bhattacharjee is the corresponding author.
* A two-sentence abstract, a figure of two translates of a lattice, an
  appendix that describes each independent check by its method, and a
  section of statements and declarations (competing interests, funding, data
  availability) after the acknowledgement of the use of AI.
* The files now carry the conjecture's name and end in `final2`, the paper
  is also in `preprintMusin24CellConjectureTwoTranslates/` (the only file in
  that folder), and `MasterMusin24CellConjectureTwoTranslates_masterzipfinal2.zip`
  holds every file of the repository with the built files.
* The mathematics and the certificates are those of v1.2.0 below; v1.2.0
  was prepared but never released, so its changes reach Zenodo with v2.0.0.

### Changes of v1.2.0 (not released on its own)

* A second formal proof of Proposition 5.1, in Lean 4 with Mathlib
  (`verification/lean_kernel/`). Every step is checked by the Lean kernel:
  there is no `native_decide`, and the main theorem `D4.certificate` depends
  only on the axioms `propext`, `Classical.choice` and `Quot.sound`. For the
  ten vertices and all 1137 edges it proves membership in the polyhedron,
  the minimal vectors, the values of det J and det Q, that each vertex is a
  vertex and that the listed rays span its cone, the sign of det J or
  det Q − 256 along every edge, the end points of the bounded edges up to
  the integral symmetries, and that the unbounded edges stay in the
  polyhedron. Run `verification/lean_kernel/check.sh`, or
  `verification/shell/run_all.sh --kernel`.
* The paper says so in the abstract, the introduction, the proof of
  Proposition 5.1 and the list of checks in the appendix.

v1.1.1 added the run logs of `crosscheck/` and `analysis/`, and moved the
agreement of the ten vertices with Andreanov and Kallus into a footnote.
v1.1.0 added the manuscript in the form submitted to Discrete & Computational
Geometry, a copy of the d4-voronoi-cells v2.0.0 code
([doi:10.5281/zenodo.23240354](https://doi.org/10.5281/zenodo.23240354)), the
floating-point cross-check of (G) and (C) on 943 two-periodic packings, and
two analytic approaches to (G) and (C) with the points where each stops.

### Files

* `Musin24CellConjectureTwoTranslates_final2.pdf`: the paper (13 pages, amsart)
* `Musin24CellConjectureTwoTranslates_final2-tex.zip`: LaTeX source with the figures as PNG
* `Musin24CellConjectureTwoTranslates_final2-arxiv.tar.gz`: LaTeX source with the figures as PDF

### Verification

The one computational statement, the vertex and edge table of the polyhedron
of two-periodic packings (Proposition 5.1), is exact and re-checked by four
independent programs: C (integers, including all extreme rays from
14-subsets of minimal vectors), Julia (rationals, Sturm sequences), Lean 4
(kernel proof of the sign lemma and evaluation of every certificate) and
Python (sympy). Run `verification/shell/run_all.sh`. A fifth check,
`verification/lean_kernel/`, proves the same table in Lean 4 with Mathlib with
every step checked by the kernel.

### Citation

Deep Bhattacharjee and Ushashi Bhattacharya. Concept DOI (all versions):
[10.5281/zenodo.23239767](https://doi.org/10.5281/zenodo.23239767).
Version 2.0.0: [10.5281/zenodo.23387006](https://doi.org/10.5281/zenodo.23387006).
Version 1.1.1: [10.5281/zenodo.23288720](https://doi.org/10.5281/zenodo.23288720).
Version 1.1.0: [10.5281/zenodo.23242482](https://doi.org/10.5281/zenodo.23242482).

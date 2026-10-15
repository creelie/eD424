# Densest packings of two translates of a lattice in four dimensions

**Deep Bhattacharjee** (corresponding author) and **Ushashi Bhattacharya**

This repository holds the paper and all code and certificates for the
following theorem.

* Paper (PDF): [`preprintMusin24CellConjectureTwoTranslates/Musin24CellConjectureTwoTranslates.pdf`](preprintMusin24CellConjectureTwoTranslates/Musin24CellConjectureTwoTranslates.pdf)
* Everything in one file: [MasterMusin24CellConjectureTwoTranslates_masterzipfinal2.zip](https://github.com/creelie/eD424/raw/main/MasterMusin24CellConjectureTwoTranslates_masterzipfinal2.zip)
* Lean formalization, programs and certificates on Zenodo: [doi:10.5281/zenodo.23387006](https://doi.org/10.5281/zenodo.23387006) (version 2.0.0; all versions: [doi:10.5281/zenodo.23239767](https://doi.org/10.5281/zenodo.23239767))

> **Theorem.** Let Λ ⊂ R⁴ be a lattice and b ∈ R⁴ \ Λ such that distinct
> points of P = Λ ∪ (Λ + b) are at distance at least 2. Then covol(Λ) ≥ 16,
> with equality only when P is congruent to √2 D₄. Equivalently, no packing
> of unit balls whose centres form two translates of a lattice is denser
> than the D₄ lattice packing (density π²/16).

**Corollary.** In such a packing every Voronoi cell has volume at least 8,
the volume of the regular 24-cell, with equality only for √2 D₄. This is
Musin's 24-cell conjecture for packings whose centres form at most two
translates of a lattice.

Scope: the method is specific to two translates. The 24-cell conjecture for
arbitrary packings, and the question whether D₄ is the densest packing of
unit balls in R⁴, remain open and are not claimed here.

## Proof outline

* The packing condition is a locally finite family of linear inequalities on
  the 5 × 5 Gram matrix J of a lattice basis and b; they cut out a
  polyhedron R (Section 2).
* Concavity of log det Q and of the Schur complement of Q in J moves every
  packing, without increasing det Q, to a vertex of R or to a point of an
  edge of R (Proposition 3.2).
* Up to the integral symmetries, R has ten vertices and 1137 edges at them
  (Proposition 5.1). This finite table is the only statement that rests on
  computation; it is exact, re-checked by four independent programs, and
  proved in Lean 4 with Mathlib, every step checked by the kernel (no
  `native_decide`; axioms `propext`, `Classical.choice`, `Quot.sound` only).
* Bounded edges carry no packings in their interiors; on the 25 unbounded
  edges everything is explicit (Lemma 5.3, Table 5.3). The minimum 256 of
  det Q is attained at two points, both √2 D₄, by a covering lemma for two
  index-two sublattices of D₄ (Lemma 4.2).

## Layout

| Path | Contents |
| --- | --- |
| `paper/main.tex`, `paper/figures/` | the paper (amsart) and its TikZ figures |
| `preprintMusin24CellConjectureTwoTranslates/` | the paper as a PDF, the only file in the folder |
| `dist/` | `Musin24CellConjectureTwoTranslates_final2.pdf`, `_final2-tex.zip` (figures as PNG) and `_final2-arxiv.tar.gz` |
| `MasterMusin24CellConjectureTwoTranslates_masterzipfinal2.zip` | every file of the repository with `dist/` and the preprint folder |
| `journal/JOURNAL.md` | the target journal (Discrete & Computational Geometry): scope, format and AI policy |
| `verification/python/ryshkov2.py` | exact enumeration of R (rationals, cddlib in GMP) |
| `verification/python/certify.py` | integer certificates in `verification/data/` |
| `verification/python/edges.py` | candidate points and det Q at each (sympy) |
| `verification/c/check_polyhedron.c` | minimal vectors, all extreme rays from 14-subsets, edges, orbits (int128) |
| `verification/julia/check_candidates.jl` | candidates and det Q at each (Sturm sequences, rationals) |
| `verification/lean/R4Check.lean` | Lean 4 kernel proof of the sign lemma; evaluation of every certificate |
| `verification/lean_kernel/` | Proposition 5.1 in Lean 4 with Mathlib, checked by the kernel; see its README |
| `verification/data/` | the vertex and edge tables, see `FORMAT.md` |
| `verification/shell/run_all.sh` | runs every check |
| `d4-voronoi-cells/` | the code for the Voronoi cell bound for all packings and the open statements (G) and (C); see its README |
| `crosscheck/` | (G) and (C) tested on two-translate packings with both codebases |
| `analysis/` | two routes to (G) and (C) by inequalities, and where each stops |

## Reproduce

```sh
scripts/build_paper.sh                 # dist/, the preprint folder and the master zip
scripts/build_submission.sh            # build/submission: PDF and source zip (Fig1 to Fig3 as PDF and EPS)
verification/shell/run_all.sh          # C, Julia, Lean 4 and Python checks
verification/shell/run_all.sh --full   # also recompute R from scratch
verification/shell/run_all.sh --kernel # also build the kernel-checked Lean proof
verification/lean_kernel/check.sh      # only the kernel-checked Lean proof
```

The checks need gcc, Python 3 with sympy (and pycddlib for `--full`),
Julia ≥ 1.6 and Lean 4 (tested with 4.9.0; no Mathlib). `JULIA` and `LEAN`
may point to binaries that are not on `PATH`. The kernel-checked proof needs
elan and pycddlib; it uses Lean 4.34.1 with Mathlib, fetched by `lake`.

## Citation

Deep Bhattacharjee and Ushashi Bhattacharya, *Densest packings of two
translates of a lattice in four dimensions*, Zenodo, 2026.
[doi:10.5281/zenodo.23239767](https://doi.org/10.5281/zenodo.23239767)
(all versions; v2.0.0 is
[doi:10.5281/zenodo.23387006](https://doi.org/10.5281/zenodo.23387006), v1.1.1 is
[doi:10.5281/zenodo.23288720](https://doi.org/10.5281/zenodo.23288720), v1.1.0 is
[doi:10.5281/zenodo.23242482](https://doi.org/10.5281/zenodo.23242482) and v1.0.0 is
[doi:10.5281/zenodo.23239768](https://doi.org/10.5281/zenodo.23239768)).
See also `CITATION.cff`.

## License

MIT, see `LICENSE`. The files in `d4-voronoi-cells/` are under
`d4-voronoi-cells/LICENSE`.

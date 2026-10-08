# d4-voronoi-cells: code for the Voronoi cell bound in four dimensions

This directory holds the computations behind the statement that every Voronoi
cell of a packing of unit balls in R^4 has volume at least 8, the volume of
the cell of D4 (the regular 24-cell of circumradius sqrt2). It is the code of
the v2.0.0 source package (concept DOI
[10.5281/zenodo.22766562](https://doi.org/10.5281/zenodo.22766562)) without
its manuscript, release notes and split archive. Labels such as
`sec:closure` or `thm:count30` in the READMEs below are LaTeX labels of that
manuscript. The use of the code is governed by `LICENSE` in this directory.

`../crosscheck/` tests the statements below on the two-translate packings of
the paper in `../paper/`, with this code and the code in `../verification/`.

## The two open statements

Scale the centres so that the minimum distance is 2 and put one centre at the
origin. Let Y be the set of the other centres y with |y| < sqrt6, and

    T(Y) = 9 pi^2 / 8 - U(Y),
    U(Y) = sum_y S(|y|) - sum_{y,y'} Pi(|y|/2, |y'|/2, <y,y'>/|y||y'|),

the volume of the cell cut off by the ball of radius sqrt(3/2) (no three of
the caps cut off by the bisectors meet inside that ball, so the sum stops at
pairs). `multi_cap/truncated_search.py` computes S, Pi and T in closed form.
The bound vol >= 8 for every cell follows from

- **(G)** if |Y| = 24 and T(Y) <= 8, then the part of the cell inside the
  convex hull of 0 and the inverted points 4y/|y|^2 has volume at least 8,
  with equality only for sqrt2 times a root system of type D4;
- **(C)** if |Y| >= 25, then T(Y) > 8, that is U(Y) < 9 pi^2/8 - 8 = 3.10330.

The cases |Y| <= 23 and |Y| >= 31 are proved, and so are (C) for 29 and 30
centres and for 28 centres when at most 14 lie within 2.0161 (15 with at most
three beyond 2.35, 16 with none beyond). (G), (C) at 25, 26 and 27 centres,
and the rest of (C) at 28 are open. `gap_closure/README.md` records where
the semidefinite bounds stop: on sampled constraints their optima at 25, 26
and 27 centres are above the level 3.10330, so no certificate of that kind
exists there, and at 28 they level off near 3.095.

## Layout

| path | what it does |
| --- | --- |
| `core/` | shared geometry: the high-precision cell volume, Sturm sequences |
| `cap_certificate/` | the cap inequality, from exact vertex enumeration to the final bound |
| `arc1_v1w1/`, `arc2_w1v2/` | the certificates on the two boundary arcs of the deviation domain |
| `swap_configs/` | exact feasibility and volume of the swap configurations |
| `hessian_multidir/` | joint Hessians along several directions (numerical observations only) |
| `multi_cap/` | the contact and non-contact cases, the cardinality certificates and the exact checks of (C) at 28 to 30 centres |
| `gap_closure/` | floating-point programmes for the open cases, with the three runs that gave certificates |
| `level2/`, `zonal/` | the second-level Lasserre programme of de Laat, Leijenhorst and de Muinck Keizer, rebuilt to run on an ordinary machine |
| `third_party/llm24-certificate/` | fetches and checks their published certificate |
| `lean/` | Lean 4 checks of finite arithmetic and polynomial identities |
| `verification/`, `independent_verification/` | end-to-end checks and the record of an independent re-run |
| `data/` | cached tables the scripts read |
| `misc/` | small supporting computations |

## Requirements

Python 3.10 or later with numpy, scipy, sympy and mpmath; cvxpy (with SCS or Clarabel) for the semidefinite
programmes; python-flint for the exact interval checks; Julia 1.10 for
`level2/`; Lean `leanprover/lean4:v4.34.0-rc2` (pinned in
`lean/lean-toolchain`) for `lean/`.

## Running

    python3 multi_cap/truncated_search.py rays   # T at the root system and along two rays
    python3 cap_certificate/cap_inequality_certificate.py
    cd lean && ./run_all.sh                      # every Lean file, with axiom report

The exact checks at 28 to 30 centres are in `multi_cap/` (`combo30_check.py`,
`combo_case_check.py`); each takes a case file from `gap_closure/` and a
certificate from `multi_cap/radial_certificates/`. They need several hours on
four cores.

The archive of de Laat, Leijenhorst and de Muinck Keizer (145 MB, data
doi:10.4121/74ce1c25-6fca-4680-8a36-e9c18e7e9594) is not copied here;
`third_party/llm24-certificate/fetch_certificate.py` downloads it and checks
its MD5.

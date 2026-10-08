# The second level on an ordinary machine

This directory lets the code of

> D. de Laat, N. M. Leijenhorst, W. H. H. de Muinck Keizer,
> *Optimality and uniqueness of the D4 root system*, arXiv:2404.18794,
> data: 4TU.ResearchData, doi:10.4121/74ce1c25-6fca-4680-8a36-e9c18e7e9594

set up and solve its second-level programme without the construction of the
zonal matrices that its README puts at three days and 128 GB.  The zonal
matrices computed in `../zonal` (about two hours, under 400 MB) are written in
the file format that code reads, and it takes them as its own.

It also compares the two constructions directly, which `../zonal` could not
do: entries computed from scratch by the authors' `compute_PS` agree with ours
coefficient by coefficient.

## The files

| file | what it does |
| --- | --- |
| `zonalstore_4.tar.gz` | the 490 entries of P(S) for n = 4, in the authors' cache format (150 KB) |
| `pkl2cache.py` | writes that cache from `ps.txt.reduced.pkl`, the reduced output of `../zonal`; the tarball is its output |
| `setup_cache.sh` | installs the cache into a copy of the authors' package |
| `las2_slack.jl` | the second-level bound on the enlarged domain [-1, 1/2 + s], for several s |
| `las2_margin.jl` | the same programme with a margin in the two-point constraint (below) |
| `xcheck_zonal.jl` | evaluates entries of Z_lambda with the authors' `evaluate_zonal_matrix` from the installed cache |
| `their_entries.jl` | runs the authors' `compute_PS` one signature at a time, in a clean folder |
| `compare_entries.py` | compares the entries it writes with ours, in exact rationals |
| `runs/` | the logs |
| `fast/` | the programme at (14, 16) on this machine: chunked build, Schur complement in triple-double arithmetic, and the programme with exactly 24 points (see `fast/README.md`) |
| `EXTERNAL_RUN.md` | what a larger machine would still add |

## Running it

Install Julia 1.10 and the authors' package as its README says
(`LasserreSphericalCodes.zip` from the 4TU record, then `Pkg.instantiate()`),
and from its folder:

```
sh /path/to/level2/setup_cache.sh .
julia --project=. -t 4 /path/to/level2/las2_slack.jl 8 10 128 B.txt 0 1//200
```

`setup_cache.sh` also writes empty placeholders for the integrand files
`cache/zonalstore/pol-2-*.txt`; `compute_PS` looks for those before it looks
for the entries themselves, and with them present it prepares nothing.

## The two constructions agree

`xcheck_zonal.jl` evaluates eight entries of Z_lambda, with |lambda| up to 14,
at the inner products (1/3, -2/7, 1/5, -1/4, 2/9, 3/11) through the authors'
code reading our cache, and `xcheck_zonal.py` finds them equal to the values
of `../zonal/zonal.py`, including one reached through the authors' exchange of
the two indices (`runs/xcheck_zonal.log`).  So the format is read as intended.

The entries themselves can be compared as well.  The authors' construction
needs its 128 GB for all signatures at once; run one signature at a time it
fits in a few gigabytes for the smaller ones.  `compare_entries.py` reads the
files written by their `compute_PS`, run from scratch in a clean folder, and
compares them with ours.  It was run for thirteen signatures, (2, 0), (3, 1),
(4, 0), (4, 2), (5, 1), (5, 3), (6, 2), (6, 4), (7, 1), (8, 0), (8, 2), (9, 1)
and (10, 0), and every one of their 88 entries is identical, coefficient by
coefficient (`runs/compare_entries.log`).  The signatures with larger lambda_2
cost more: (10, 2) passed 3 GB, and was stopped there.

## What the programme gives here

At slack 0, on a machine with four cores and 16 GB:

| degrees (d1, delta) | bound | iterations | time | memory |
| --- | --- | --- | --- | --- |
| (4, 6) | 32 | | 100 s in all | |
| (8, 10) | 26.0000 | 68 | 42 min on one core | 3.5 GB |
| (10, 12) | 24.9423 | 62 | 3.9 h on four cores | 11.8 GB |

(10, 12) is the most that fits in 16 GB with the authors' solver as it is:
from (8, 10) the time of an iteration grows about eightfold and the memory
more than threefold.  With the chunked build and the faster Schur complement
of `fast/`, (14, 16) runs in 10.3 GB, at about a quarter of an hour per
iteration on four cores; at slack 1/125 it bounds the number of points by
24.5555 (floating point).  The
certificate of the authors uses (14, 16), where the bound is 24.  At slack 0
every degree reached here is weaker than the three-point bound, 24.13.  The
logs are in `runs/`.

## A margin in the two-point constraint

`las2_margin.jl` fixes the bound K(empty, empty) at a number N and maximises
mu subject to

    A_2K({x, y}) + SOS_2(u) + mu w(u) = 0,
    w(u) = (u + 1) (u + 1/2)^2 u^2 (1/2 + s - u),

which is nonnegative on [-1, 1/2 + s]; the other constraints are those of the
authors.  For a code of 24 points with inner products in that interval the
chain of their proof then gives mu times the sum of w over the pairs at most
N - 24, so every inner product has w(u) <= (N - 24)/mu, and lies close to
-1, -1/2 or 0, or within s of 1/2.  That is the form of the classification
of 24 points at positive slack that the localisation of the paper's last
section would need.  Near the double
zeros w(u) is about u^2/8, so the window is about (8 (N - 24)/mu)^(1/2): the
programme turns a bound N - 24 of order s into a window of order s^(1/2)
unless the bound grows more slowly than s.

Here it runs at (4, 6) with s = 0 and N = 33, where it finds mu = 2.2134
(`runs/las2_margin_4_6_s0_N33.log`); at that degree the bound itself is 32,
(N - 24)/mu = 4.07 exceeds the largest value of w on the interval, and nothing
is pinned.

**It pins nothing at any degree once s > 5.2e-4.**  The normalised roots of
D4 are themselves a 24-point code with inner products in [-1, 1/2 + s], and w
equals 3s/8 on each of their 96 pairs at 1/2, so every feasible (N, mu) has
(N - 24)/mu >= 36 s.  At s = 0.008 that is 0.288, while w never exceeds
0.019779 on the interval: the inequality w(u) <= (N - 24)/mu excludes no inner
product, whatever the degree and whatever the machine.  `margin_floor_check.py`
checks this (exactly, and in ball arithmetic).  A weight that also vanishes
at 1/2 escapes this floor, but no pointwise reading can beat the codes that
exist, and at s = 0.008 there are 24-point codes with an inner product of
0.1231, far from every root value.  So the (14, 16) margin run is not worth
doing.


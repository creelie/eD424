# The second level at (14, 16) on a 16 GB machine

The programme of `../README.md` at the degrees (d1, delta) = (14, 16) of the
certificate of de Laat, Leijenhorst and de Muinck Keizer, on the enlarged
domain [-1, 1/2 + s] with s = 1/125, solved on four cores and 16 GB.  Two
changes make it fit: the programme is built in chunks, each written to disk
as soon as it is built, and the Schur complement, the step that dominates
each iteration, is formed from double-precision matrix products by exact
splitting and accumulated in triple-double arithmetic, on a 128-bit working
precision.

## Results (floating point)

| programme | value | iterations | time per iteration | memory |
| --- | --- | --- | --- | --- |
| number of points, s = 1/125 | 24.5555 (gap 1.3e-5) | 86 | about 15 min on 4 cores | 10.3 GB |
| (6, 8), same programme, check | 26.7286 | 51 | 18 s on 1 core | 1.8 GB |
| (6, 8), exactly 24 points, weighted defect | 4.37987 | 54 from scratch, 27 warm | 16 s on 1 core | 1.9 GB |
| (14, 16), exactly 24 points, weighted defect | at least about 1.44 (stopped) | 5, warm | about 14 min on 4 cores | 11 GB |

The value 24.5555 bounds the number of points of slack 1/125 in the
relaxation; it is evidence, not a theorem, since no certificate is rounded
and checked.  `analyze_cert.jl` reads the two-point multiplier P2 off the
solution: it is below 1e-4 at inner products up to 0.3 and below 0.004
everywhere (`runs/P2_1416_it86.txt`), so the certificate charges almost
nothing to any single pair.

The programme with exactly 24 points (`solve_fc.jl`) maximises the weighted
design defect sum_k c_k S_k, with the weights c_k of an earlier version of
the paper; a value below 0.26785 at (14, 16) would bound the defect of
every 24-point code of slack 1/125, once rounded and checked.  At (6, 8) its
value is 4.38, which says nothing at that degree.  Fixing all four subset
counts leaves no interior (the variance of the number of points is then 0),
and the solver's dual diverges (`runs/fc_6_8_eq_diverged.log`); the
programme therefore fixes the number of points and bounds the number of
pairs by binomial(24, 2) + FC_EPS.

At (14, 16), with FC_EPS = 1/100, the programme gives nothing either.  It was
started warm from iteration 64 of the plain run (the log's "iteration 51"
counts from that run's resumption at iteration 13); the machine restarted
twice and the run resumed from its checkpoints (`runs/fc_1416_part1_it1-3.log`,
`runs/fc_1416_part2_it4-5.log`, rows renumbered from 1 at each resumption).
Its moment side read 1.439 after iterations 4 and 5 while the relative
infeasibility fell from 3.1e-3 to 3.6e-4, and the run was stopped there.
`diag_fc_primal.jl` reads the pair side of that moment solution: 276.57 pairs
and S_1, ..., S_5 = 0.24, 0.24, 0.46, 0.73, 6.41
(`runs/fc_1416_it5_moments.txt`).  For these moments no test polynomial of
degree up to 11 excludes a further centre (`hole_test.py`,
`runs/hole_test_fc_1416.txt`), while real codes of slack 1/125 found by local
search reach only 0.09 in the weighted defect.

## The files

| file | what it does |
| --- | --- |
| `build.sh`, `build_part.jl` | build the chunk files (about 1.8 GB for (14, 16)) |
| `assemble.jl` | assembles the chunks into one programme |
| `assemble_fc.jl` | the same with exactly FC_N points and the defect objective |
| `solve.sh`, `solve_fast.jl` | solve the programme, with a checkpoint after every iteration |
| `solve_fc.sh`, `solve_fc.jl` | solve the programme with exactly 24 points |
| `patched_solver.jl` | the solver of ClusteredLowRankSolver 1.0.3 with checkpoints and warm starts (its licence: `LICENSE.ClusteredLowRankSolver`) |
| `tiled_S.jl`, `fast_S.jl` | the Schur complement, tiled and in split double-precision products |
| `analyze_cert.jl` | reads the bound and the multiplier P2 off a checkpoint, and checks P2 against the constraint matrices |
| `diag_fc_primal.jl` | reads the pair side (pair count and S_k) of the moment solution of the programme with exactly 24 points |
| `hole_test.py` | tests whether any polynomial of degree up to 11 excludes a further centre, given S_1, S_2, ... |
| `counts_check.jl` | sums the moments by subset size (the number of points, pairs, triples, quadruples) |
| `runs/` | the logs |

## Running it

Install the authors' package and the zonal cache as in `../README.md`, then

```
LSC=/path/to/LasserreSphericalCodes sh build.sh b1416
LSC=/path/to/LasserreSphericalCodes sh solve.sh b1416 solve.log ckpt.jls
julia --project=$LSC analyze_cert.jl b1416 ckpt_copy.jls 16 1//125
```

`analyze_cert.jl` also writes `b1416.names.jls`, the block names that a warm
start of `solve_fc.sh` needs.

## The run, and what to watch

The starting scale matters.  A first run from 1e3 stalled at iteration 70,
with the dual objective falling below 24 before the dual was feasible
(`runs/stalled_omega1e3_part*.log`); from 1e10 both sides were feasible at
iteration 61 and the gap was below 1e-3 at iteration 84.  The run was
interrupted by restarts of the machine and resumed from its checkpoints:
`runs/solve_1416_part1..6*.log` are its pieces, in order, with rows
renumbered from 1 at each resumption, and `runs/trajectory_1416.txt` lists
iteration, mu, dual objective, dual error and the two step lengths.  The
Schur complement is very ill-conditioned (smallest pivot squared over the
diagonal about 1e-34); triple-double accumulation on 128 bits kept its
error near 1e-38.  The solve takes 10.3 GB, so on a 16 GB machine nothing
large should run beside it.

A warm start of the exactly-24 programme from a checkpoint of the plain one
works: at (6, 8), from the plain run's iteration 34 (mu 3e8, both sides
feasible) it reaches the value of the cold run, 4.37987, in 27 iterations
instead of 54 (`runs/fc_6_8_warm.log`, `runs/fc_6_8_cold.log`,
`runs/plain_6_8_omega1e10.log`).

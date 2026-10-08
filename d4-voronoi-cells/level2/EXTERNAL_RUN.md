# What a larger machine would add

**The programmes at the degrees (14, 16) now run on this machine** (four
cores, 16 GB): the bound on the number of points of slack 1/125, 24.5555,
and the programme with exactly 24 points, whose value is at least about
1.44, far above the 0.26785 the design defects would need, both in `fast/`
(see `fast/README.md`).  The estimate made earlier from (8, 10) and (10, 12), about
135 GB and twelve days for (14, 16) with the authors' own solver, no longer
applies to them.

**The margin (step 4 of the earlier plan) gives nothing at slack 0.008, on
any machine.**  The root system spends the margin: its 96 pairs at inner
product 1/2 force (N - 24)/mu >= 36 s = 0.288, while the weight never exceeds
0.019779 (`margin_floor_check.py`).  `las2_margin.jl` is kept for the record.

What a larger machine would still add is precision.  A value of these
programmes becomes a theorem only through a certificate rounded to rationals
and checked, as the authors did for theirs at 256 bits.  The runs here work
at 128 bits; a 256-bit run of (14, 16) needs roughly twice the memory and
several times the time of an iteration (an estimate, not measured).  The
commands are those of `fast/README.md` with the working precision 256 in
place of 128 in `solve.sh` or `solve_fc.sh`.

## The authors' own solver, for comparison

With the zonal cache installed (`setup_cache.sh`), the authors' solver runs
the same programme through `las2_slack.jl`:

    julia --project=. -t 8 /path/to/level2/las2_slack.jl 14 16 128 B.txt 0 1//125

Measured here: (8, 10) in 42 minutes on one core and 3.5 GB, (10, 12) in
3.9 hours on four cores and 11.8 GB; (14, 16) does not fit in 16 GB this
way.

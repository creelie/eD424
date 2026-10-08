# bigmachine: runs that need more memory than the cloud container

The cloud container has 15 GB of memory and 4 cores. The programmes here are
run on a larger machine and write their logs next to themselves.

## run_typed.py: twenty-four close centres and one more

The typed three-point programme of `multi_cap/typed_cardinality_sdp.py`: 24
points of S^3 with pairwise inner products at most t1 = 0.508 (centres within
2.0161) and a further point with inner product at most t2 with each of them.
A corrected Z below 0 says that no such code exists.

| t2 | radius | what an exclusion gives |
|---|---|---|
| 0.6141 | sqrt6 | statement (C) whenever 24 centres lie within 2.0161: no further centre fits within sqrt6 |
| 0.51135 | 2.03 | the count {24 within 2.0161, at most 24 within 2.03} at 26 centres, one of the two leaves of the density split in `../README.md` |

At degree 14 in the container the sampled Z at t2 = 0.6141 is +0.018
(`../density/typed2_d14_6141.log`); degrees 16 and 18 are run here.

    python -m pip install --user numpy scipy cvxpy clarabel
    python gap_closure/bigmachine/run_typed.py 16 18 --t2 0.6141

The log `run_typed.log` records the machine, the package versions and one line
per round; a copy of the run on record is `records/run_typed_DESKTOP-VS7DF9S_2026-10-05.log`.
At t2 = 0.6141 degree 16 ends its four rounds at sampled Z +0.00248 and degree 18
starts at +0.00005: no certificate, the values levelling off at zero. Floating point on sampled constraints: a corrected Z below 0 is what
an exact check would then have to confirm, and no result here is used in a
proof until it has one.

## run_combo.py: the combined kernel of statement (C), several counts at once

`../CM/combo_direct.py` (the two-point kernel labelled by distance plus the
three-point kernel on directions typed by distance range, handed to Clarabel
directly) for the case files of `../CM`, one process per job:

    python gap_closure/bigmachine/run_combo.py case29_all.json:10:4 case27_all.json:8:5

Each job writes `<tag>_d<degree>.log` here and one certificate per round to
`../CM` (tag with suffix `pc`); `run_combo.log` records the machine and the
last round of each job. A corrected bound below 9 pi^2/8 - 8 = 3.10330 is then
checked exactly by `multi_cap/combo_case_check.py`.

# rebuilt_from_text

`d4_independent_check.py` recomputes the numerical claims that the remaining
case of the paper depends on from the statements in the text alone, without
importing anything from the package, so that agreement is a second witness
and not a shared bug.  It needs mpmath, numpy and scipy, and runs in a few
minutes:

    python3 d4_independent_check.py

It checks the covering bounds (pi m / 3) tan^3 r_m of Table 1, the four
values of the pair bound after the second-order proposition, the spectrum of
the rigidity operator and the diagonal 11/16 of the projection onto its image
(Lemma 7.50), the crossings 0.155 and 0.197 of the pair terms near the root
system, and the crossing 0.147 when the truncation radius is grown with the
distances.  Its output is `d4_independent_check.log`.  Floating point and
quadrature at 30 digits; the exact versions are in multi_cap/.

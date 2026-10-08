/-
KissStat.lean: the runs of KissMain8 and KissMain10 reported as pairs
(status, number of intervals or boxes processed), status 0 meaning that
the run finished with every interval or box verified or discarded, and
the bound B = 1 + f(1) + F(1,1,1) as a rational number.  A record, not
a theorem.
-/
import KissChecks

#eval (cert8.bound checked8.F, checked8.statI fuelI)
#eval (cert10.bound checked10.F, checked10.statI fuelI)
#eval checked8.statII fuelII
#eval checked10.statII fuelII

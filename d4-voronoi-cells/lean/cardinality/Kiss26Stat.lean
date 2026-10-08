/-
Kiss26Stat.lean: the run of Kiss26Main reported as pairs (status, number of
intervals or boxes processed), status 0 meaning that the run finished with
every interval or box verified or discarded, and the bound
B = 1 + f(1) + F(1,1,1) as a rational number.  A record, not a theorem.
-/
import Kiss26Main

#eval (cert26.bound checked26.F, checked26.statI fuelI)
#eval checked26.statII fuelII

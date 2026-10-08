/-
D4LabelledStat.lean: the runs of D4LabelledMain reported as pairs (status,
boxes), status 0 meaning that the branch and bound finished with every box
verified or discarded.  A record, not a theorem.
-/
import D4LabelledDomain

#eval statSlab 20000000
#eval statGamma 20000000

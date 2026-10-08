/-
D4CertStat.lean: the same run as D4CertMain, reported as a pair (status,
boxes): status 0 means the branch and bound finished with every box
verified or discarded, and the second entry is the number of boxes it
processed.  This is a record, not a theorem.
-/
import D4CertDomain

#eval verifyStat 4000000

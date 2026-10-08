/-
KissChecks.lean: the two certificates prepared for the check (P1 and P2
expanded, scaled, differentiated), and the fuel of the branch and bounds.
Compiled with the rest of the library, so that native_decide runs the
checks as native code.
-/
import KissDomain

def checked8 : Checked := prepare cert8
def checked10 : Checked := prepare cert10

/-- Upper limits on the number of intervals and boxes the runs may process. -/
def fuelI : Nat := 1000000
def fuelII : Nat := 40000000

/-- Constraint (i) of a prepared certificate: every interval of [-1, top] verified. -/
def Checked.okI (k : Checked) : Bool := (k.statI fuelI).1 == 0
/-- Constraint (ii): every box of the ordered admissible domain in [-1, top]^3 verified or discarded. -/
def Checked.okII (k : Checked) : Bool := (k.statII fuelII).1 == 0

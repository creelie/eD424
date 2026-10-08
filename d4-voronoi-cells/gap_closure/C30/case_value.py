"""floating-point value of the two-point case programme (radial_case_sdp.value) for given cases.
usage: case_value.py M 'r:lo:hi,r:lo:hi' ..."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'multi_cap'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'multi_cap'))
from fractions import Fraction as Fr
import radial_case_sdp as RC
M = int(sys.argv[1])
for spec in sys.argv[2:]:
    case = {}
    for item in spec.split(','):
        r, lo, hi = item.split(':')
        case[Fr(r)] = (int(lo), int(hi))
    v = RC.value((M, case))
    print(M, spec, '%.5f' % v, 'closed' if v < RC.TARGET - 1e-3 else 'open', flush=True)

import os as _os, sys as _sys
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA_DIR = _os.path.join(_PKG_ROOT, "data")
for _p in (_PKG_ROOT, _os.path.join(_PKG_ROOT, "core")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
def _data(_name):
    """Resolve a bundled data file, wherever the script is run from."""
    _c = _os.path.join(_DATA_DIR, _name)
    if _os.path.exists(_c):
        return _c
    if _os.path.exists(_name):
        return _name
    raise SystemExit(
        "required data file %r not found; expected it in %s. "
        "Run the matching *_derive.py step first, or fetch the bundled "
        "copy from the package's data/ directory." % (_name, _DATA_DIR))

import sys, pickle
sys.path.insert(0, '.')
from common import *

# Region A (TYPE1, 32 verts): t in (0, t2=t*=arctan(1/2)), lower=W-curve,
# upper = theta=pi/3 for t<t1=0.169918, Y-curve for t1<t<t2
t1 = 0.169918
t0 = 0.08
theta0 = 0.5*(theta_W(t0) + THETA_U)

samples = [(0.02, 0.5*(theta_W(0.02)+THETA_U)),
           (0.05, 0.5*(theta_W(0.05)+THETA_U)),
           (0.1, 0.5*(theta_W(0.1)+THETA_U)),
           (0.15, 0.5*(theta_W(0.15)+THETA_U))]

vol_fixed, total = derive_region(theta0, t0, "regionA", samples, expect_verts=32)
print("vol_fixed:", vol_fixed)
with open(_os.path.join(_DATA_DIR, 'regionA_data.pkl'), 'wb') as f:
    pickle.dump({'vol_fixed': str(vol_fixed), 'total': str(total), 'theta0': theta0, 't0': t0}, f)

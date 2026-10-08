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

# Region C (TYPE3, 34 verts): lower switches Y(t<t1)->U(t1<t<t3)->W(t>t3), upper=A-curve throughout
t1, t3 = 0.169918, 0.615480
t0 = 0.1
theta0 = 0.5*(theta_Y(t0) + theta_A(t0))

samples = [(0.05, 0.5*(theta_Y(0.05)+theta_A(0.05))),
           (0.15, 0.5*(theta_Y(0.15)+theta_A(0.15))),
           (0.3, 0.5*(THETA_U+theta_A(0.3))),
           (0.55, 0.5*(THETA_U+theta_A(0.55))),
           (0.7, 0.5*(theta_W(0.7)+theta_A(0.7))),
           (0.78, 0.5*(theta_W(0.78)+theta_A(0.78)))]

vol_fixed, total = derive_region(theta0, t0, "regionC", samples, expect_verts=34)
print("vol_fixed:", vol_fixed)
with open(_os.path.join(_DATA_DIR, 'regionC_data.pkl'), 'wb') as f:
    pickle.dump({'vol_fixed': str(vol_fixed), 'total': str(total), 'theta0': theta0, 't0': t0}, f)

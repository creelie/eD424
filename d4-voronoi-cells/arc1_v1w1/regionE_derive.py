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

# Region E (TYPE5, 37 verts): lower=Y-curve throughout, upper=W-curve(t2<t<t3)->U=pi/3(t>t3)
t2, t3 = 0.463648, 0.615480
t0 = 0.55
theta0 = 0.5*(theta_Y(t0) + theta_W(t0))

samples = [(0.48, 0.5*(theta_Y(0.48)+theta_W(0.48))),
           (0.52, 0.5*(theta_Y(0.52)+theta_W(0.52))),
           (0.65, 0.5*(theta_Y(0.65)+THETA_U)),
           (0.75, 0.5*(theta_Y(0.75)+THETA_U))]

vol_fixed, total = derive_region(theta0, t0, "regionE", samples, expect_verts=37)
print("vol_fixed:", vol_fixed)
with open(_os.path.join(_DATA_DIR, 'regionE_data.pkl'), 'wb') as f:
    pickle.dump({'vol_fixed': str(vol_fixed), 'total': str(total), 'theta0': theta0, 't0': t0}, f)

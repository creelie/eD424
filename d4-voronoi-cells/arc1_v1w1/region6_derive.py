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

# Region 6 (TYPE6, 36 verts): t in (t3, pi/4), lower=U=pi/3 (constant), upper=W-curve
t3 = 0.615480
t0 = 0.7
theta0 = 0.5*(THETA_U + theta_W(t0))

samples = [(0.62, 0.5*(THETA_U+theta_W(0.62))),
           (0.68, 0.5*(THETA_U+theta_W(0.68))),
           (0.75, 0.5*(THETA_U+theta_W(0.75))),
           (0.78, 0.5*(THETA_U+theta_W(0.78)))]

vol_fixed, total = derive_region(theta0, t0, "region6", samples, expect_verts=36)
print("vol_fixed:", vol_fixed)
with open(_os.path.join(_DATA_DIR, 'region6_data.pkl'), 'wb') as f:
    pickle.dump({'vol_fixed': str(vol_fixed), 'total': str(total), 'theta0': theta0, 't0': t0}, f)

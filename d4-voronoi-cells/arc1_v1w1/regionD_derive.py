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

# Region D (TYPE4, 37 verts): t in (t1,t3), lower=Y-curve(t1<t<t2)->W-curve(t2<t<t3), upper=U=pi/3
t1, t2, t3 = 0.169918, 0.463648, 0.615480
t0 = 0.3
theta0 = 0.5*(theta_Y(t0) + THETA_U)

samples = [(0.2, 0.5*(theta_Y(0.2)+THETA_U)),
           (0.35, 0.5*(theta_Y(0.35)+THETA_U)),
           (0.5, 0.5*(theta_W(0.5)+THETA_U)),
           (0.6, 0.5*(theta_W(0.6)+THETA_U))]

vol_fixed, total = derive_region(theta0, t0, "regionD", samples, expect_verts=37)
print("vol_fixed:", vol_fixed)
with open(_os.path.join(_DATA_DIR, 'regionD_data.pkl'), 'wb') as f:
    pickle.dump({'vol_fixed': str(vol_fixed), 'total': str(total), 'theta0': theta0, 't0': t0}, f)

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

import sympy as sp
import numpy as np
import pickle, time

t = sp.symbols('t', real=True, positive=True)
with open(_data('swap_exact_verts_rational.pkl'),'rb') as f:
    exact_verts = pickle.load(f)
simplices = np.load(_data('swap_simplices_theta03.npy'))

t0val = np.tan(0.15)

t0 = time.time()
vol_terms = []
for i, simp in enumerate(simplices):
    M = sp.Matrix.hstack(*[exact_verts[idx] for idx in simp])
    d = sp.cancel(M.det())
    vol_terms.append(d)
    if i % 20 == 0:
        print(f"  simplex {i}/{len(simplices)}  t={time.time()-t0:.1f}s")
print(f"all determinants computed in {time.time()-t0:.1f}s")

with open(_os.path.join(_DATA_DIR, 'swap_vol_terms.pkl'), 'wb') as f:
    pickle.dump(vol_terms, f)

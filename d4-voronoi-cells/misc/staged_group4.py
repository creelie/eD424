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
with open(_data('swap_vol_terms.pkl'),'rb') as f:
    vol_terms = pickle.load(f)
with open(_data('swap_quads_theta03.pkl'),'rb') as f:
    qdata = pickle.load(f)
quad_for_vertex = qdata['quad_for_vertex']
simplices = np.load(_data('swap_simplices_theta03.npy'))

depends = [any(idx in (22,23) for idx in q) for q in quad_for_vertex]
t0val = np.tan(0.15)

group4 = []
for i, simp in enumerate(simplices):
    ndep = sum(depends[idx] for idx in simp)
    if ndep == 4:
        dnum = float(vol_terms[i].subs(t, t0val))
        sign = 1 if dnum > 0 else -1
        group4.append(sign*vol_terms[i])

print(f"group 4: {len(group4)} terms", flush=True)
t0=time.time()
raw_sum = sum(group4, sp.Integer(0))
print(f"summed raw in {time.time()-t0:.1f}s", flush=True)

t0=time.time()
simplified = sp.cancel(raw_sum)
print(f"cancel() done in {time.time()-t0:.1f}s", flush=True)
val = float(simplified.subs(t, t0val))
print(f"numeric@t0val={val:.6f}", flush=True)

with open(_os.path.join(_DATA_DIR, 'group4_total.pkl'), 'wb') as f:
    pickle.dump(simplified, f)
print("saved.", flush=True)

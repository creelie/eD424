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
import pickle, time

t = sp.symbols('t', real=True, positive=True)
with open(_data('group_totals_0to3.pkl'),'rb') as f:
    g0to3 = pickle.load(f)
with open(_data('group4_total.pkl'),'rb') as f:
    g4 = pickle.load(f)

all_groups = [g0to3[0], g0to3[1], g0to3[2], g0to3[3], g4]

t0=time.time()
raw = sum(all_groups, sp.Integer(0))
print(f"summed 5 group totals in {time.time()-t0:.1f}s", flush=True)

t0=time.time()
combined = sp.cancel(raw)
print(f"cancel() of combined done in {time.time()-t0:.1f}s", flush=True)

# this is 24*vol; get F_swap = combined/24 - 8
t0=time.time()
F_swap = sp.cancel(combined/24 - 8)
print(f"F_swap simplification done in {time.time()-t0:.1f}s", flush=True)

print()
print("F_swap(t) =")
print(F_swap)
print()
print("size (chars):", len(str(F_swap)))

with open(_os.path.join(_DATA_DIR, 'F_swap_closed_form.pkl'), 'wb') as f:
    pickle.dump(F_swap, f)
print("saved.")

# validate at known exact points
t0val = 0.15  # placeholder, will use exact rationals below
import numpy as np
for tt, expected in [(sp.Rational(1,20), 0.00975691), (sp.Rational(1,8), 0.05785433),
                      (sp.Rational(1,4), 0.20571816)]:
    val = float(F_swap.subs(t, tt))
    print(f"t={tt}: F_swap={val:.8f}  expected~{expected}  match={abs(val-expected)<1e-5}")

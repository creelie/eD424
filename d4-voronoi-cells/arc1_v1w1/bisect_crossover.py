#!/usr/bin/env python3
"""
bisect_crossover.py

Locates, by bisection on an exact symbolic difference, the crossover
parameter at which the swap-path and mixed-path defect totals of
sec:swap-not-worst exchange order.  Each evaluation is a full symbolic
simplification over Q(sqrt 3) and takes a few seconds, so the default
forty bisections run for about three minutes on ordinary consumer
hardware.  Set BISECT_ITERATIONS to shorten that.
The exact value it converges to is the one reported in the paper, and
crossover_exact.py reproduces that value directly without the
bisection.

Requires crossover_totals.pkl, produced by the swap-configuration
derivation and shipped in the package's data/ directory.  Set
CROSSOVER_PKL to override the location.
"""
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

import os
import pickle
import time

import numpy as np
import sympy as sp

_PKL = os.environ.get("CROSSOVER_PKL") or _data("crossover_totals.pkl")

ITERATIONS = int(os.environ.get("BISECT_ITERATIONS", "40"))

t = sp.symbols('t', real=True, positive=True)
with open(_PKL, 'rb') as f:
    data = pickle.load(f)
total_swap = data['total_swap']
total_mixed = data['total_mixed']


def diffF(tt):
    Fs = sp.cancel(sp.radsimp(total_swap.subs(t, tt)))/24
    Fm = sp.cancel(sp.radsimp(total_mixed.subs(t, tt)))/24
    return float(Fs - Fm)

lo, hi = sp.Rational(33,100), sp.Rational(35,100)
dlo, dhi = diffF(lo), diffF(hi)
assert dlo>0 and dhi<0

t0=time.time()
for it in range(ITERATIONS):
    mid = (lo+hi)/2
    dm = diffF(mid)
    if dm > 0:
        lo = mid
    else:
        hi = mid
    print(it, float(mid), dm, round(time.time()-t0,1), flush=True)

tc = (lo+hi)/2
thetac = 2*sp.atan(tc)
print('t_c (rational bisection) =', float(tc))
print('theta_c =', sp.N(thetac, 30))
with open(_os.path.join(_DATA_DIR, 'theta_c_bisect.pkl'), 'wb') as f:
    pickle.dump({'lo':lo,'hi':hi}, f)

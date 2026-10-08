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

STAGE_SECONDS = float(_os.environ.get("SWAPVOL_STAGE_SECONDS", "300"))


class _Timeout(Exception):
    pass


def _bounded(seconds, fn, *args):
    """Run one symbolic call under a wall-clock bound.

    cancel() and simplify() on the assembled sum do not always settle
    here; this is the expression growth documented in the methodology
    appendix.  Bounding the call lets the script report what it reached
    rather than run without end.
    """
    import signal

    def _fire(signum, frame):
        raise _Timeout()

    old = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return True, fn(*args)
    except _Timeout:
        return False, None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
import numpy as np
import pickle, time

t = sp.symbols('t', real=True, positive=True)
with open(_data('swap_vol_terms.pkl'),'rb') as f:
    vol_terms = pickle.load(f)

t0val = np.tan(0.15)
t0 = time.time()
total = sp.Integer(0)
for i, d in enumerate(vol_terms):
    dnum = float(d.subs(t, t0val))
    sign = 1 if dnum > 0 else -1
    total += sign*d
print(f"summed (unsimplified) in {time.time()-t0:.1f}s")

t0=time.time()
ok, total_c = _bounded(STAGE_SECONDS, sp.cancel, total)
if not ok:
    print(f"cancel() did not settle within {STAGE_SECONDS:.0f}s; the closed"
          f" form for this configuration is obtained instead by the"
          f" per-term route of swap_exact_volume2.py")
    raise SystemExit(0)
print(f"cancel() done in {time.time()-t0:.1f}s")

t0=time.time()
ok, total_simpl = _bounded(STAGE_SECONDS, sp.simplify, total_c)
if not ok:
    print(f"simplify() did not settle within {STAGE_SECONDS:.0f}s; the"
          f" cancelled form below is the result")
    total_simpl = total_c
else:
    print(f"simplify() done in {time.time()-t0:.1f}s")
print("total volume*24 (raw, cancel):")
print(total_c)
print()
print("total (simplified):")
print(total_simpl)

with open(_os.path.join(_DATA_DIR, 'swap_total_vol.pkl'), 'wb') as f:
    pickle.dump(total_c, f)

# numeric check
val = float(total_c.subs(t, t0val))/24
print()
print("numeric check: vol(0.3) =", val, "  (expect ~8.0827528)")

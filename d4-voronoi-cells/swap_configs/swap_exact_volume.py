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


def _bounded(seconds, fn, *args, **kw):
    """Run one symbolic call under a wall-clock bound.

    simplify() on the assembled signed sum does not always settle here;
    this is the expression growth documented in the methodology appendix.
    Bounding the call lets the script save the raw sum, which is what the
    later stages actually consume, instead of running without end.
    """
    import signal

    def _fire(signum, frame):
        raise _Timeout()

    old = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return True, fn(*args, **kw)
    except _Timeout:
        return False, None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
import numpy as np
import pickle, time

theta = sp.symbols('theta', real=True, positive=True)
with open(_data('swap_exact_verts.pkl'),'rb') as f:
    exact_verts = pickle.load(f)
simplices = np.load(_data('swap_simplices_theta03.npy'))
verts_num = np.load(_data('swap_verts_theta03.npy'))
print("n simplices:", len(simplices))

t0 = time.time()
vol_terms = []
for i, simp in enumerate(simplices):
    M = sp.Matrix.hstack(*[exact_verts[idx] for idx in simp])  # 4x4
    d = M.det()
    vol_terms.append(d)
    if i % 40 == 0:
        print(f"  simplex {i}/{len(simplices)}  t={time.time()-t0:.1f}s")
print(f"all determinants computed in {time.time()-t0:.1f}s")

# determine correct sign for each simplex numerically at theta=0.3 (orientation), then sum |det|/24
t0=time.time()
signed_vol = sp.Integer(0)
for i, d in enumerate(vol_terms):
    # Orientation only.  Substituting theta into the symbolic determinant
    # is fragile here: on a few of these expressions the substitution
    # leaves an unevaluated form that will not reduce to a number.  The
    # numerical vertices at the same theta are already loaded and give the
    # orientation directly, which is what the rest of the package does.
    sign = 1 if np.linalg.det(verts_num[list(simplices[i])]) > 0 else -1
    signed_vol += sign*d
print(f"signed sum assembled, t={time.time()-t0:.1f}s")

t0=time.time()
ok, _simpl = _bounded(STAGE_SECONDS, sp.simplify, signed_vol)
if ok:
    signed_vol_simplified = sp.nsimplify(_simpl, [sp.sqrt(2), sp.sqrt(3), sp.sqrt(6)])
    print(f"simplify done t={time.time()-t0:.1f}s")
    print("signed_vol (pre /24):", signed_vol_simplified)
else:
    print(f"simplify() did not settle within {STAGE_SECONDS:.0f}s; saving the"
          f" raw signed sum, which is what the later stages consume")

with open(_os.path.join(_DATA_DIR, 'swap_signed_vol_raw.pkl'), 'wb') as f:
    pickle.dump(signed_vol, f)

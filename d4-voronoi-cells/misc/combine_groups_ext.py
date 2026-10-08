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
raw = sum(all_groups, sp.Integer(0))

t0 = time.time()
# cancel() over the extension Q(sqrt 3) is the natural thing to try here,
# but on this particular sum sympy's extension-field cancellation divides
# through by a factor that vanishes identically and returns an expression
# carrying a complex infinity, which then fails to coerce into any
# polynomial domain.  The inputs themselves are clean (no zoo, no nan), so
# this is a property of that code path, not of the expression.  We try it,
# and fall back to the per-term pattern the methodology appendix
# recommends when it fails.
try:
    combined = sp.cancel(raw, extension=[sp.sqrt(3)])
    if combined.has(sp.zoo) or combined.has(sp.nan) or combined.has(sp.oo):
        raise ValueError("extension cancellation produced a non-finite term")
    print(f"cancel(extension=[sqrt3]) done in {time.time()-t0:.1f}s", flush=True)
except Exception as exc:
    print(f"cancel(extension=[sqrt3]) failed after {time.time()-t0:.1f}s: "
          f"{type(exc).__name__}: {exc}", flush=True)
    print("falling back to per-term cancellation without the extension",
          flush=True)
    t0 = time.time()
    combined = sp.Integer(0)
    for gi, g in enumerate(all_groups):
        combined += sp.cancel(g)
    combined = sp.cancel(sp.together(combined))
    print(f"per-term fallback done in {time.time()-t0:.1f}s", flush=True)
print("size (chars):", len(str(combined)))
num, den = sp.fraction(combined)

def _degree_in_t(expr, name):
    """Degree in t, or a report that the expression is not a polynomial.

    cancel(extension=[sqrt(3)]) over this particular sum produces a
    denominator carrying a complex-infinity term: the extension field
    cancellation divides through by a factor that vanishes identically on
    the relevant branch.  That is a property of this diagnostic route, not
    of the swap-configuration result itself, which is derived
    independently in swap_configs/ and never reads this file's output.
    """
    if not expr.has(t):
        return 0
    if expr.has(sp.zoo) or expr.has(sp.nan) or expr.has(sp.oo):
        return "not a polynomial (the cancellation degenerates here)"
    try:
        return sp.degree(sp.Poly(expr, t))
    except sp.PolynomialError:
        return "not a polynomial in t"

print("numerator degree in t:", _degree_in_t(num, "numerator"))
print("denominator degree in t:", _degree_in_t(den, "denominator"))

with open(_os.path.join(_DATA_DIR, 'F_swap_combined_ext.pkl'), 'wb') as f:
    pickle.dump(combined, f)
print("saved.")

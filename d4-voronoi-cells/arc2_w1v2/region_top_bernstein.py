import os as _os
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA_DIR = _os.path.join(_PKG_ROOT, "data")
def _data(_name):
    """Resolve a bundled or cached data file, wherever the script is run from."""
    _os.makedirs(_DATA_DIR, exist_ok=True)
    return _os.path.join(_DATA_DIR, _name)

import pickle, time
import sympy as sp
from math import comb

w, v2 = sp.symbols('w v2', real=True, positive=True)
sqrt2 = sp.sqrt(2)

data = pickle.load(open(_data('region_top_ND.pkl'),'rb'))
N = sp.sympify(data['N'], locals={'w':w,'v2':v2})
D = sp.sympify(data['D'], locals={'w':w,'v2':v2})

def poly_to_coeff_dict(expr):
    coeffs = {}
    for term in expr.as_ordered_terms():
        pd = term.as_powers_dict()
        i, j = int(pd.get(w,0)), int(pd.get(v2,0))
        c = term
        if i: c = c/w**i
        if j: c = c/v2**j
        c = sp.nsimplify(sp.expand(c), [sp.sqrt(2)])
        coeffs[(i,j)] = coeffs.get((i,j), sp.Integer(0)) + c
    return coeffs

t0 = time.time()
Nc = poly_to_coeff_dict(N)
Dc = poly_to_coeff_dict(D)
print(f"N: {len(Nc)} terms, D: {len(Dc)} terms, {time.time()-t0:.1f}s")
nw = max(i for i,j in Nc); nvN = max(j for i,j in Nc)
dw = max(i for i,j in Dc); dvD = max(j for i,j in Dc)
print(f"N degree: (w={nw}, v2={nvN})   D degree: (w={dw}, v2={dvD})")

with open(_data('region_top_coeffs.pkl'),'wb') as f:
    pickle.dump({'Nc': {k: str(vv) for k,vv in Nc.items()}, 'Dc': {k: str(vv) for k,vv in Dc.items()},
                 'nw': nw, 'nvN': nvN, 'dw': dw, 'dvD': dvD}, f)
print("saved coeffs.")

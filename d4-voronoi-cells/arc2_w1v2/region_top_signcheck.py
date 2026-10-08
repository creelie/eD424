import os as _os
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA_DIR = _os.path.join(_PKG_ROOT, "data")
def _data(_name):
    """Resolve a bundled or cached data file, wherever the script is run from."""
    _os.makedirs(_DATA_DIR, exist_ok=True)
    return _os.path.join(_DATA_DIR, _name)

import pickle
import sympy as sp
w, v2 = sp.symbols('w v2', real=True, positive=True)
data = pickle.load(open(_data('region_top_ND.pkl'),'rb'))
N = sp.sympify(data['N'], locals={'w':w,'v2':v2})
D = sp.sympify(data['D'], locals={'w':w,'v2':v2})
for (wv,v2v) in [(0.5,0.5),(0.1,0.1),(0.9,0.9),(0.5,0.1),(0.5,0.9)]:
    nval = float(N.subs({w:wv, v2:v2v}))
    dval = float(D.subs({w:wv, v2:v2v}))
    print(f"w={wv} v2={v2v}: N={nval:.6f}  D={dval:.6f}  N/D={nval/dval:.6f}")

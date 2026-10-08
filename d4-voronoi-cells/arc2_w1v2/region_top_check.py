import os as _os
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA_DIR = _os.path.join(_PKG_ROOT, "data")
def _data(_name):
    """Resolve a bundled or cached data file, wherever the script is run from."""
    _os.makedirs(_DATA_DIR, exist_ok=True)
    return _os.path.join(_DATA_DIR, _name)

import pickle, numpy as np, sympy as sp
theta, t = sp.symbols('theta t', real=True, positive=True)
data = pickle.load(open(_data('region_top_data.pkl'),'rb'))
total = sp.sympify(data['total'], locals={'theta':theta,'t':t})
expr = total - sp.Rational(5,3)
f = sp.lambdify((theta,t), expr, 'numpy')

def theta_A(tt): return 2*np.arctan(np.cos(tt))
minval = 1e9
for tt in np.linspace(0.001, np.pi/4-0.001, 60):
    thlo = theta_A(tt)+0.001
    for th in np.linspace(thlo, np.pi/2-0.001, 60):
        val = f(th,tt)
        if val < minval:
            minval = val; argmin=(th,tt)
print("min of vol_moving_top - 5/3 over grid:", minval, "at", argmin)

# check theta->pi/2 edge and theta->theta_A edge values
for tt in [0.01, 0.1, 0.3, 0.6, 0.77]:
    thlo = theta_A(tt)
    print(f"t={tt:.3f}: at theta_A={thlo:.4f}: f={f(thlo+1e-6,tt):.6f}   at pi/2: f={f(np.pi/2-1e-6,tt):.6f}")

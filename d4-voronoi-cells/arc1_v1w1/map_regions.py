import numpy as np
from scipy.optimize import brentq

def theta_W(t): return 2*np.arctan(np.sin(t))
def theta_A(t): return 2*np.arctan(np.cos(t))
def theta_Y(t):
    val = 1.0/(np.sin(t)+np.cos(t))
    val = min(val, 1.0)
    return np.arcsin(val)
theta_U = np.pi/3

curves = {'W': theta_W, 'A': theta_A, 'Y': theta_Y, 'U': lambda t: theta_U}

t_lo, t_hi = 1e-6, np.pi/4 - 1e-6

# find all pairwise crossings within (t_lo, t_hi)
names = list(curves.keys())
crossings = []
tt = np.linspace(t_lo, t_hi, 20000)
for i in range(len(names)):
    for j in range(i+1, len(names)):
        f = lambda t: curves[names[i]](t) - curves[names[j]](t)
        vals = np.array([f(x) for x in tt])
        sign_changes = np.where(np.diff(np.sign(vals)) != 0)[0]
        for idx in sign_changes:
            a, b = tt[idx], tt[idx+1]
            try:
                root = brentq(f, a, b)
                crossings.append((root, names[i], names[j], curves[names[i]](root)))
            except Exception as e:
                pass

crossings.sort()
print("Crossings (t, curve1, curve2, theta):")
for c in crossings:
    print(f"  t={c[0]:.6f} ({np.degrees(c[0]):.3f} deg)  {c[1]}={c[2]}  theta={c[3]:.6f} ({np.degrees(c[3]):.3f} deg)")

# Now scan a fine grid and for each t, print the ordering of curve values (only those in (0,pi/2))
print()
print("Ordering of curves by theta at sample t values:")
for t in [0.02, 0.1, 0.169918, 0.2, 0.3, 0.4636, 0.5, 0.6, 0.7, np.pi/4-0.01]:
    vals = sorted([(curves[n](t), n) for n in names])
    print(f"  t={t:.6f} ({np.degrees(t):.2f} deg): " + ", ".join(f"{n}={np.degrees(v):.2f}" for v,n in vals))

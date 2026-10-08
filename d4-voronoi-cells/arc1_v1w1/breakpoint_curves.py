import numpy as np

# The 12 nontrivial (a,b,c) classes found (excluding (0,0,0) and (-1,0,0)):
classes = [
    (-0.5,-0.5,-0.5), (-0.5,-0.5,0.5), (-0.5,0.5,-0.5), (-0.5,0.5,0.5),
    (0.0,-1.0,0.0), (0.0,0.0,-1.0), (0.0,0.0,1.0), (0.0,1.0,0.0),
    (0.5,-0.5,-0.5), (0.5,-0.5,0.5), (0.5,0.5,-0.5), (0.5,0.5,0.5),
]

# Equation: cos(theta)*a + sin(theta)*(b*cos(t)+c*sin(t)) = 1
# => R(t)*cos(theta - phi(t)) = 1, R(t)=sqrt(a^2+g(t)^2), phi(t)=atan2(g(t),a)
# where g(t) = b*cos(t)+c*sin(t). Real solution theta in (0,pi/2) exists
# iff R(t) >= 1, giving theta = phi(t) +- arccos(1/R(t)) (mod domain).

ts = np.linspace(0.0, np.pi/2, 13)
print(f"{'t':>7}", end="")
for (a,b,c) in classes:
    print(f" | ({a:+.1f},{b:+.1f},{c:+.1f})", end="")
print()

curve_data = {cls: [] for cls in classes}
for t in ts:
    row = f"{t:7.3f}"
    for (a,b,c) in classes:
        g = b*np.cos(t)+c*np.sin(t)
        R = np.sqrt(a*a+g*g)
        phi = np.arctan2(g,a) if a != 0 or g != 0 else None
        if R < 1.0 - 1e-9 or phi is None:
            row += f" |     --     "
            curve_data[(a,b,c)].append(None)
            continue
        cosval = 1.0/R
        cosval = min(1.0, max(-1.0, cosval))
        delta = np.arccos(cosval)
        theta1 = phi - delta
        theta2 = phi + delta
        cands = [th for th in (theta1,theta2) if 0 < th < np.pi/2]
        if cands:
            th = min(cands)
            row += f" |  {th:8.4f}  "
            curve_data[(a,b,c)].append(th)
        else:
            row += f" |     --     "
            curve_data[(a,b,c)].append(None)
    print(row)

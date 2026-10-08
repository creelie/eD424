# Check ||G - G0||_F^2 >= 12 F^2 - 4 sqrt(6) F^3 (F = Procrustes distance) on the extremal codes found.
import numpy as np, itertools, glob
R = []
for i, j in itertools.combinations(range(4), 2):
    for a in (1, -1):
        for b in (1, -1):
            v = np.zeros(4); v[i] = a; v[j] = b; R.append(v / np.sqrt(2))
R = np.array(R)
for f in glob.glob('maxfrob_best_*.npy'):
    X = np.load(f); U, S_, Vt = np.linalg.svd(X.T @ R); Q = U @ Vt; F = np.linalg.norm(X - R @ Q.T)
    G = X @ X.T; G0 = R @ R.T; g2 = np.sum((G - G0) ** 2)
    print('%s: F = %.5f  ||G-G0||^2 = %.5f  lower bound 12F^2-4sqrt6F^3 = %.5f  (linear 12F^2 = %.5f)' % (f, F, g2, 12*F**2 - 4*np.sqrt(6)*F**3, 12*F**2))
F = 0.22782; print('at F* = %.5f: 12F^2 - 4 sqrt6 F^3 = %.5f -> need sum_{i<j} (u-r)^2 < %.5f' % (F, 12*F**2 - 4*np.sqrt(6)*F**3, (12*F**2 - 4*np.sqrt(6)*F**3) / 2))

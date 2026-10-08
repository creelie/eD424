"""
swap_defect_curves.py -- the volume defect along the three swap paths,
computed directly from the cell.

The three finite-angle results of the swap section are proved from exact
closed forms.  This script recomputes the same three functions by an
independent route, enumerating the local Voronoi region of the deviated
configuration with scipy and taking its volume, and writes the curves so
that the figure in that section shows measured values rather than a
schematic.  For the hexagon the closed form is available in the text and
is evaluated alongside, as a check on the identification of the path.

Also computed: the swap path against the mixed path, whose crossover the
section reports near theta = 0.65.

Writes swap_curves.dat in the figures directory of the paper and prints
the agreement of the two routes for the hexagon.
"""
import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull

TARGET = 8.0
THETA_STAR = np.pi / 3


def root_system():
    R = []
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(4)
                    v[i] = si
                    v[j] = sj
                    R.append(v / np.sqrt(2))
    return np.array(R)


ROOTS = root_system()
G = ROOTS @ ROOTS.T


def cell_volume(dirs):
    hs = np.hstack([dirs, -np.ones((len(dirs), 1))])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    return ConvexHull(hi.intersections, qhull_options='QJ').volume


def perp(a, b):
    """the unit vector in the plane of a and b, orthogonal to a, on b's side"""
    w = b - (a @ b) * a
    return w / np.linalg.norm(w)


def cyclic_path(idx):
    """u_i(theta) = cos(theta) a_i + sin(theta) (a_{i+1} perp a_i)"""
    A = ROOTS[idx]
    n = len(idx)
    P = np.array([perp(A[i], A[(i + 1) % n]) for i in range(n)])
    rest = ROOTS[[k for k in range(24) if k not in idx]]

    def at(theta):
        U = np.cos(theta) * A + np.sin(theta) * P
        return np.vstack([rest, U])
    return at


def mixed_path(idx):
    """the same pair with the sign of the first deviation reversed"""
    A = ROOTS[idx]
    P = np.array([perp(A[0], A[1]), perp(A[1], A[0])])
    P[0] = -P[0]
    rest = ROOTS[[k for k in range(24) if k not in idx]]

    def at(theta):
        U = np.cos(theta) * A + np.sin(theta) * P
        return np.vstack([rest, U])
    return at


def find_clique(size):
    """indices of `size` roots, consecutive ones at inner product 1/2, the
    cycle closing up; for size 2 and 3 this is a clique"""
    import itertools
    for c in itertools.combinations(range(24), size):
        ok = all(abs(G[c[i], c[(i + 1) % size]] - 0.5) < 1e-9
                 for i in range(size))
        if size <= 3:
            ok = ok and all(abs(G[a, b] - 0.5) < 1e-9
                            for a, b in itertools.combinations(c, 2))
        if ok:
            return list(c)
    raise SystemExit('no cycle of size %d' % size)


def hex_cycles(limit=400):
    """closed walks of six distinct roots, consecutive ones at inner
    product 1/2, up to rotation of the cycle"""
    adj = [[j for j in range(24) if abs(G[i, j] - 0.5) < 1e-9]
           for i in range(24)]
    seen, out = set(), []
    def walk(path):
        if len(out) >= limit:
            return
        if len(path) == 6:
            if path[0] in adj[path[-1]]:
                key = min(tuple(path[k:] + path[:k]) for k in range(6))
                key = min(key, min(tuple(path[::-1][k:] + path[::-1][:k])
                                   for k in range(6)))
                if key not in seen:
                    seen.add(key)
                    out.append(list(path))
            return
        for j in adj[path[-1]]:
            if j > path[0] and j not in path:
                walk(path + [j])
    for s in range(24):
        walk([s])
    return out


def hex_closed(theta):
    t = np.tan(theta / 2)
    A = 27 * t**8 - 108 * t**6 + 90 * t**4 + 100 * t**2 + 3
    B = 48 * t**5 - 96 * t**3 - 16 * t
    den = (t**2 - 3)**2 * (t**2 + 1)**2 * (3 * t**2 - 1)**2
    return 8 * t**4 * (A + np.sqrt(3) * B) / den


def sample(at, ts):
    out = []
    for th in ts:
        try:
            out.append(cell_volume(at(th)) - TARGET)
        except Exception:
            out.append(np.nan)
    return np.array(out)


if __name__ == '__main__':
    pair = find_clique(2)
    tri = find_clique(3)
    print('pair  %s' % pair)
    print('triangle %s' % tri)

    probe = np.array([0.20, 0.35, 0.50])
    want = hex_closed(probe)
    hexa, best = None, np.inf
    for c in hex_cycles():
        at = cyclic_path(c)
        try:
            got = np.array([cell_volume(at(th)) - TARGET for th in probe])
        except Exception:
            continue
        err = float(np.max(np.abs(got - want)))
        if err < best:
            best, hexa = err, c
        if err < 1e-7:
            break
    print('hexagon  %s, closest match to the closed form %.2e' % (hexa, best))

    ts = np.linspace(0.0, THETA_STAR, 211)
    f2 = sample(cyclic_path(pair), ts)
    f3 = sample(cyclic_path(tri), ts)
    f6 = sample(cyclic_path(hexa), ts)
    fc = hex_closed(np.maximum(ts, 1e-9))

    good = np.isfinite(f6) & (ts > 0.02) & (ts < THETA_STAR - 0.02)
    print('hexagon: enumerated against closed form, max |difference| %.2e'
          % np.max(np.abs(f6[good] - fc[good])))
    print('defect at theta=0: swap %.2e, triangle %.2e, hexagon %.2e'
          % (f2[0], f3[0], f6[0]))
    print('least defect on the open interval: swap %.6f, triangle %.6f, '
          'hexagon %.6f'
          % (np.nanmin(f2[1:-1]), np.nanmin(f3[1:-1]), np.nanmin(f6[1:-1])))

    tm = np.linspace(0.05, 1.5, 200)
    sw = sample(cyclic_path(pair), tm)
    mx = sample(mixed_path(pair), tm)
    d = sw - mx
    k = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    cross = [round(float(tm[i] - d[i] * (tm[i + 1] - tm[i])
                         / (d[i + 1] - d[i])), 4) for i in k]
    print('swap against mixed: crossings at %s' % cross)
    print('at theta=0.05: swap %.6f, mixed %.6f' % (sw[0], mx[0]))

    with open('swap_curves.dat', 'w') as f:
        f.write('theta swap tri hex\n')
        for i, th in enumerate(ts):
            f.write('%.6f %.8f %.8f %.8f\n' % (th, f2[i], f3[i], f6[i]))
    with open('swap_mixed.dat', 'w') as f:
        f.write('theta swap mixed\n')
        for i, th in enumerate(tm):
            f.write('%.6f %.8f %.8f\n' % (th, sw[i], mx[i]))
    print('wrote swap_curves.dat and swap_mixed.dat')

#!/usr/bin/env python3
"""
symmetric_search.py -- saturated 23-point configurations with a symmetry.

A contact configuration that is invariant under a nontrivial element of
O(4) is a union of orbits of that element, and the search for a saturated
one can then be carried out in a space of far fewer parameters, with
enough starts to cover it well.  This script runs the search over every
cyclic rotation type of order n <= 12 and over the two improper
involutions, one orbit structure at a time.

  Rotation types.  A rotation of order n through the angles 2 pi a / n
  and 2 pi b / n in two orthogonal planes P1, P2, with gcd(a, b, n) = 1.
  A point off both planes has an orbit of size n; a point of P1 has an
  orbit of size n1 = n / gcd(a, n), a point of P2 one of size
  n2 = n / gcd(b, n).  An invariant configuration of 23 directions is
  k generic orbits with s1 orbits in P1 and s2 in P2, so that
  k n + s1 n1 + s2 n2 = 23; the parameters are k points of S^3 (three
  parameters each) and s1 + s2 angles.  Points on the same great circle
  P1 are pairwise 60 degrees apart only when there are at most six of
  them, which bounds s1 n1 and s2 n2.

  Improper involutions.  A reflection in a hyperplane, whose fixed set
  is a great 2-sphere (orbits of size 1 with two parameters, and of size 2
  with three), and the element diag(1,-1,-1,-1), whose fixed set is a
  pair of antipodal points.

For each orbit structure the inradius g(W) of the convex hull of the
resulting configuration is maximised, subject to the contact constraints
on all 253 pairs, by sequential least-squares programming on the epigraph
form with the facet list refreshed between solves, from a number of
random starts.  The output records, per structure, the largest g reached
by a feasible endpoint and whether the endpoint is a deletion of a root
(inner-product multiset -1, -1/2, 0, 1/2 with multiplicities 11, 88, 66,
88).  The whole run is exploration, and it says so.

Usage: python3 symmetric_search.py [starts per structure] [seed]
"""
import sys, time, math, itertools
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

M = 23
IU = np.triu_indices(M, 1)
DELETION = {-1.0: 11, -0.5: 88, 0.0: 66, 0.5: 88}

def rot(n, a, b):
    t1, t2 = 2*math.pi*a/n, 2*math.pi*b/n
    R = np.eye(4)
    R[0,0], R[0,1], R[1,0], R[1,1] = math.cos(t1), -math.sin(t1), math.sin(t1), math.cos(t1)
    R[2,2], R[2,3], R[3,2], R[3,3] = math.cos(t2), -math.sin(t2), math.sin(t2), math.cos(t2)
    return R

def sph(p):
    """three parameters -> unit vector in R^4 (not surjective onto the great
    circles P1, P2, which the special orbits supply separately)."""
    a, b, c = p
    return np.array([math.cos(a)*math.cos(b), math.cos(a)*math.sin(b),
                     math.sin(a)*math.cos(c), math.sin(a)*math.sin(c)])

class Structure:
    """An orbit structure: powers of a matrix g applied to parametrised reps."""
    def __init__(self, name, G, gens):
        # G: list of group elements (matrices); gens: list of ('gen', npar) ...
        self.name = name; self.G = G; self.reps = gens
        self.npar = sum(npar for _, npar in gens)
    def build(self, params):
        pts = []; k = 0
        for kind, npar in self.reps:
            p = params[k:k+npar]; k += npar
            if kind == 'generic':
                v = sph(p)
            elif kind == 'P1':
                v = np.array([math.cos(p[0]), math.sin(p[0]), 0, 0])
            elif kind == 'P2':
                v = np.array([0, 0, math.cos(p[0]), math.sin(p[0])])
            elif kind == 'S2':          # fixed 2-sphere x4 = 0
                v = np.array([math.cos(p[0])*math.cos(p[1]), math.cos(p[0])*math.sin(p[1]), math.sin(p[0]), 0])
            elif kind == 'axis':        # the fixed point e1
                v = np.array([1.0, 0, 0, 0])
            orbit = []
            for g in self.G:
                w = g @ v
                if all(np.linalg.norm(w - u) > 1e-9 for u in orbit):
                    orbit.append(w)
            pts.extend(orbit)
        W = np.array(pts)
        return W

def structures(nmax=12):
    out = []
    # rotations
    for n in range(2, nmax+1):
        for a in range(0, n):
            for b in range(a, n):
                if math.gcd(math.gcd(a, b), n) != 1: continue
                if a == 0 and b == 0: continue
                R = rot(n, a, b); G = [np.linalg.matrix_power(R, m) for m in range(n)]
                n1 = n // math.gcd(a, n) if a else 1
                n2 = n // math.gcd(b, n) if b else 1
                for k in range(0, 23 // n + 1):
                    rest = 23 - k*n
                    for s1 in range(0, 7):
                        if s1*n1 > 6 or s1*n1 > rest: continue
                        r2 = rest - s1*n1
                        if r2 % n2: continue
                        s2 = r2 // n2
                        if s2*n2 > 6: continue
                        if s1 and n1 == 1 and s1 > 6: continue
                        reps = [('generic', 3)]*k + [('P1', 1)]*s1 + [('P2', 1)]*s2
                        out.append(Structure(f"C{n}({a},{b}) k={k} P1x{s1} P2x{s2}", G, reps))
    # improper involutions
    S = np.diag([1., 1., 1., -1.]); G = [np.eye(4), S]
    for f in range(1, 8):           # f fixed points on the 2-sphere, (23 - f)/2 pairs
        if (23 - f) % 2: continue
        k = (23 - f) // 2
        out.append(Structure(f"reflection fixed={f} pairs={k}", G, [('generic', 3)]*k + [('S2', 2)]*f))
    P = np.diag([1., -1., -1., -1.]); G = [np.eye(4), P]
    out.append(Structure("diag(1,-1,-1,-1) fixed=1 pairs=11", G, [('generic', 3)]*11 + [('axis', 0)]))
    return out

def g_of(W):
    try:
        hull = ConvexHull(W)
    except Exception:
        return None, None
    return float((-hull.equations[:, -1]).min()), hull

def optimise(struct, rng, iters=6, slack=0.0, x=None):
    if x is None:
        x = rng.uniform(-math.pi, math.pi, size=struct.npar)
    def W_of(x):
        W = struct.build(x)
        if W.shape[0] != M: return None
        return W
    W = W_of(x)
    if W is None: return None
    # feasibility first: minimise the largest inner product, in epigraph form
    def worst(x):
        W = W_of(x); G = W @ W.T; np.fill_diagonal(G, -1); return G.max()
    def pairs_s(z):
        W = W_of(z[:-1]); G = W @ W.T
        return z[-1] - G[IU]
    z0 = np.r_[x, worst(x)]
    for _ in range(3):
        res = minimize(lambda z: z[-1], z0, method='SLSQP',
                       constraints=[{'type': 'ineq', 'fun': pairs_s}],
                       options={'maxiter': 300, 'ftol': 1e-13})
        z0 = res.x
        if worst(z0[:-1]) <= 0.5 + slack + 1e-9: break
    x = z0[:-1]
    if worst(x) > 0.5 + slack + 1e-7:
        return None
    # epigraph maximisation of g with facets refreshed
    for _ in range(iters):
        W = W_of(x); g, hull = g_of(W)
        if hull is None: return None
        Fs = hull.simplices
        Fs = Fs[np.abs(np.linalg.det(W[Fs])) > 1e-7]
        def offs(x, Fs=Fs):
            W = W_of(x); P = W[Fs]
            try:
                nrm = np.linalg.solve(P, np.ones((len(Fs), 4, 1)))[:, :, 0]
            except np.linalg.LinAlgError:
                return -np.ones(len(Fs))
            return 1.0 / np.linalg.norm(nrm, axis=1)
        def pairs(x):
            W = W_of(x); G = W @ W.T
            return 0.5 + slack - G[IU]
        z0 = np.r_[x, g]
        cons = [{'type': 'ineq', 'fun': lambda z: offs(z[:-1]) - z[-1]},
                {'type': 'ineq', 'fun': lambda z: pairs(z[:-1])}]
        res = minimize(lambda z: -z[-1], z0, method='SLSQP', constraints=cons,
                       options={'maxiter': 200, 'ftol': 1e-12})
        x = res.x[:-1]
    W = W_of(x); g, hull = g_of(W)
    if hull is None: return None
    G = W @ W.T; np.fill_diagonal(G, -1)
    if G.max() > 0.5 + slack + 1e-7: return None
    return g, W, G, x

def multiset(G):
    u, c = np.unique(np.round(G[IU], 6), return_counts=True)
    return dict(zip(u.tolist(), c.tolist()))

def main():
    starts = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    rng = np.random.default_rng(seed)
    t0 = time.time()
    S = structures()
    print(f"{len(S)} orbit structures, {starts} starts each")
    best_all = -1; n_feas = 0; n_sat = 0
    SLACKS = [0.05, 0.02, 0.01, 0.005, 0.002, 0.0]
    for st in S:
        best = -1; feas = 0; isdel = False; best_relaxed = -1
        for _ in range(starts):
            # continuation in the slack, as in slack_continuation.py, inside the class
            x = None; out = None
            for slack in SLACKS:
                out = optimise(st, rng, slack=slack, x=x)
                if out is None: break
                x = out[3]
                if slack == 0.01: best_relaxed = max(best_relaxed, out[0])
            if out is None: continue
            g, W, G, x = out; feas += 1
            if g > best:
                best = g; isdel = (multiset(G) == DELETION)
            if g > 0.5 + 1e-6:
                n_sat += 1
                np.save(f"symmetric_candidate_{st.name.replace(' ','_')}.npy", W)
                print(f"  !! {st.name}: g = {g:.9f} > 1/2, saved")
        n_feas += feas
        tag = "deletion" if isdel else ("none feasible" if feas == 0 else "other")
        print(f"{st.name:34s} params {st.npar:3d}  feasible {feas:3d}/{starts}  best g {best:11.8f}  "
              f"(at slack 0.01: {best_relaxed:9.6f})  {tag}", flush=True)
        best_all = max(best_all, best)
    print()
    print(f"largest g over all structures: {best_all:.9f}; feasible endpoints {n_feas}; with g > 1/2: {n_sat}")
    print("RESULT: " + ("no saturated symmetric configuration found (exploration, not proof)." if n_sat == 0
                        else "candidates saved; verify exactly."))
    print(f"total time {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()

"""
Verification of the combinatorics behind the two propositions on how much
of a root system a contact configuration can hold.

Write the roots of D_4 as (e_i +- e_j)/sqrt2 and call {i,j} the support of
such a root. The six supports fall into three couples of complementary
index pairs,

    {1,2} | {3,4},      {1,3} | {2,4},      {1,4} | {2,3},

and a support is filled for a subset S of the roots when all four of its
roots lie in S. The proofs rest on the following, all checked here.

  (i) If both supports of one couple are filled then every unit vector v
      with <v,r> <= 1/2 for all r in S is itself a root. Two removed roots
      touch at most two of the six supports, so a couple always survives.

 (ii) Three removed roots can leave no couple whole only when their three
      supports meet each couple once. There are eight such support
      patterns, four stars such as {1,2},{1,3},{1,4} and four triangles
      such as {1,2},{1,3},{2,3}, and each accounts for 4^3 = 64 triples of
      roots, so 512 of the 2024 in all.

(iii) Among those 512, the sign analysis in the proof leaves only the 96
      whose three roots are pairwise at 60 degrees. A cruder necessary
      condition, that the sum of the three roots be longer than 3/2, keeps
      192 of them, so the sign analysis is doing real work and the script
      reports both counts. The 96 form a single orbit under the symmetry
      group of the root system, with the star
      {(e_1+e_2),(e_1+e_3),(e_1+e_4)}/sqrt2 as representative.

 (iv) For that star, every direction of the doorway has first coordinate at
      least 1/sqrt2 and the other three nonnegative, and no two of them are
      as much as 60 degrees apart unless both are removed roots.

  (v) One rung lower, at four removed roots, the same question is answered
      by minimisation with a margin rather than by proof.
"""
import numpy as np
from itertools import combinations, permutations
from scipy.optimize import minimize
from scipy.spatial import HalfspaceIntersection

T = 1.0 / np.sqrt(2.0)
HALF = 0.5
COUPLES = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]


def d4_roots():
    V, sup = [], []
    for i, j in combinations(range(4), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(4)
                v[i], v[j] = si, sj
                V.append(v / np.sqrt(2.0))
                sup.append((i, j))
    return np.array(V), sup


R, SUP = d4_roots()


def group():
    gens = []
    for p in permutations(range(4)):
        P = np.zeros((4, 4))
        for a, b in enumerate(p):
            P[a, b] = 1.0
        gens.append(P)
    gens.append(np.diag([1.0, 1.0, 1.0, -1.0]))
    gens.append(0.5 * np.array([[1, 1, 1, 1], [1, 1, -1, -1],
                                [1, -1, 1, -1], [1, -1, -1, 1]], dtype=float))
    seen = {tuple(np.round(np.eye(4), 9).ravel()): np.eye(4)}
    frontier = [np.eye(4)]
    while frontier:
        nxt = []
        for M in frontier:
            for g in gens:
                N = M @ g
                k = tuple(np.round(N, 9).ravel())
                if k not in seen:
                    seen[k] = N
                    nxt.append(N)
        frontier = nxt
    out = []
    for M in seen.values():
        img = R @ M
        idx = np.argmin(np.linalg.norm(img[:, None, :] - R[None, :, :],
                                       axis=2), axis=1)
        if np.max(np.linalg.norm(img - R[idx], axis=1)) < 1e-9:
            out.append((M, tuple(int(t) for t in idx)))
    return out


G = group()
PERMS = [p for _, p in G]


def filled_supports(removed):
    hit = {SUP[i] for i in removed}
    return [s for s in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            if s not in hit]


def has_filled_couple(removed):
    comp = set(filled_supports(removed))
    return any(a in comp and b in comp for a, b in COUPLES)


def orbit_reps(k):
    seen, reps = set(), []
    for c in combinations(range(24), k):
        if c in seen:
            continue
        orb = {tuple(sorted(p[i] for i in c)) for p in PERMS}
        seen |= orb
        reps.append((c, len(orb)))
    return reps


def is_triangle(c):
    Gm = R[list(c)] @ R[list(c)].T
    return all(abs(Gm[a, b] - 0.5) < 1e-12
               for a, b in combinations(range(len(c)), 2))


if __name__ == "__main__":
    print("(i) two roots removed")
    bad = [c for c in combinations(range(24), 2) if not has_filled_couple(c)]
    print(f"    pairs of roots: {len(list(combinations(range(24), 2)))}")
    print(f"    pairs leaving no couple whole: {len(bad)}")
    print(f"    orbits of pairs under the group: {len(orbit_reps(2))}")
    print()

    print("(ii) three roots removed")
    trip = list(combinations(range(24), 3))
    bad3 = [c for c in trip if not has_filled_couple(c)]
    print(f"    triples of roots: {len(trip)}")
    print(f"    triples leaving no couple whole: {len(bad3)}")
    pats = {}
    for c in bad3:
        s = tuple(sorted(SUP[i] for i in c))
        shared = len(set(s[0]) & set(s[1]) & set(s[2]))
        kind = "star" if shared == 1 else "triangle"
        pats[kind] = pats.get(kind, 0) + 1
    print(f"    of those, by support pattern: {pats}")
    allsup = {tuple(sorted(SUP[i] for i in c)) for c in bad3}
    print(f"    distinct support patterns: {len(allsup)}"
          f"  (4 stars and 4 triangles expected)")
    print()

    print("(iii) which of them can have all three removed roots within")
    print("      60 degrees of one direction")
    print("      (the length test below is the crude necessary condition;")
    print("       the sign analysis in the proof is what leaves the 96)")
    surv = []
    for c in bad3:
        # necessary: sum of the three roots must have length above 3/2
        s = R[list(c)].sum(axis=0)
        if float(np.linalg.norm(s)) > 1.5 + 1e-12:
            surv.append(c)
    print(f"    triples passing the length test: {len(surv)}")
    print(f"    triples that are pairwise at 60 degrees: "
          f"{sum(1 for c in bad3 if is_triangle(c))}")
    print(f"    the length test alone does not isolate them: "
          f"{set(surv) == {c for c in bad3 if is_triangle(c)}}")
    print()

    tri = [c for c in trip if is_triangle(c)]
    base = None
    for c in tri:
        if sorted(SUP[i] for i in c) == [(0, 1), (0, 2), (0, 3)] and \
           all(R[i][0] > 0 for i in c) and \
           all(sum(R[i]) > 1.4 for i in c):
            base = c
            break
    print(f"    reference star triple: {base} ="
          f" {[list(np.round(R[i]*np.sqrt(2),3)) for i in base]}")
    orb = {tuple(sorted(p[i] for i in base)) for p in PERMS}
    print(f"    its orbit: {len(orb)} of {len(tri)};"
          f" transitive: {len(orb) == len(tri)}")
    print()

    print("(iv) the doorway of the reference triple, in coordinates")
    keep = np.array([i for i in range(24) if i not in base])
    K = R[keep]
    rng = np.random.default_rng(4242)
    v0 = R[list(base)].sum(axis=0)
    v0 /= np.linalg.norm(v0)
    B = np.linalg.svd(np.eye(4) - np.outer(v0, v0))[0][:, :3]
    pts = [R[list(base)]]
    for _ in range(12):
        d = rng.normal(size=(200000, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        r = 0.8 * rng.random(200000) ** (1 / 3)
        Y = v0 + (d * r[:, None]) @ B.T
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        ok = (Y @ K.T).max(axis=1) <= 0.5 + 1e-12
        if ok.any():
            pts.append(Y[ok])
    P = np.vstack(pts)
    print(f"    points found in the doorway: {len(P)}")
    print(f"    smallest first coordinate: {P[:,0].min():.12f}"
          f"   (claim: at least 1/sqrt2 = {T:.12f})")
    print(f"    smallest other coordinate: {P[:,1:].min():.12f}"
          f"   (claim: at least 0)")
    Q = P if len(P) <= 9000 else P[rng.permutation(len(P))[:9000]]
    Gm = Q @ Q.T
    np.fill_diagonal(Gm, 2.0)
    print(f"    smallest inner product between two of them:"
          f" {float(Gm.min()):.12f}   (claim: at least 1/2)")
    eq = np.argwhere(Gm < 0.5 + 1e-9)
    idset = sorted({int(i) for i, _ in eq})
    print(f"    points involved in an inner product equal to 1/2:"
          f" {len(idset)}")
    for i in idset[:6]:
        print(f"      {np.round(Q[i]*np.sqrt(2), 6)}"
              f"   is a removed root:"
              f" {bool(np.min(np.linalg.norm(R[list(base)]-Q[i], axis=1))<1e-6)}")

    print()
    print("(v) one rung lower: four removed roots")
    reps4 = orbit_reps(4)
    print(f"    orbits of four-element subsets: {len(reps4)}")
    live = []
    for c, sz in reps4:
        keep = np.array([i for i in range(24) if i not in c])
        A = np.hstack([R[keep], -np.ones((len(keep), 1))])
        P = HalfspaceIntersection(A, np.zeros(4)).intersections
        rmax = float(np.max(np.linalg.norm(P, axis=1)))
        if rmax > 2.0 + 1e-9:
            F = P[np.linalg.norm(P, axis=1) > 2.0 - 1e-9]
            F = F / np.linalg.norm(F, axis=1, keepdims=True)
            seeds = []
            for v in F:
                if all(float(v @ u) < 1 - 1e-9 for u in seeds):
                    seeds.append(v)
            live.append((c, sz, rmax, np.array(seeds)))
    print(f"    orbits whose cell has circumradius above 2: {len(live)}")
    for c, sz, rmax, seeds in live:
        G4 = R[list(c)] @ R[list(c)].T
        prof = sorted(round(float(G4[a, b]), 3)
                      for a, b in combinations(range(4), 2))
        print(f"      {c}: orbit {sz:>4}, circumradius {rmax:.9f},"
              f" mutual inner products {prof}")
    print()
    print(f"    {'removed':<15} {'margin':>7}"
          f" {'least largest pairwise inner product':>38}")
    rng4 = np.random.default_rng(90210)
    for c, sz, rmax, seeds in live:
        keep = np.array([i for i in range(24) if i not in c])
        K4 = R[keep]
        for delta in (0.0, 2.0, 5.0, 10.0):
            cosd = np.cos(np.radians(delta))
            best = 2.0
            for _ in range(400 if delta == 0.0 else 200):
                x0 = np.concatenate(
                    [seeds[int(rng4.integers(len(seeds)))]
                     + 0.35 * rng4.normal(size=4) for _ in range(3)] + [[0.9]])

                def uv(x):
                    return [x[4 * i:4 * i + 4]
                            / np.linalg.norm(x[4 * i:4 * i + 4])
                            for i in range(3)]

                def cons(x):
                    V3 = uv(x)
                    out = []
                    for v in V3:
                        out.append(HALF - K4 @ v)
                        out.append(np.array([cosd - float(np.max(R @ v))]))
                    out.append(np.array([x[12] - float(V3[a] @ V3[b])
                                         for a, b in combinations(range(3), 2)]))
                    return np.concatenate(out)

                res = minimize(lambda x: x[12], x0, method="SLSQP",
                               constraints=[{"type": "ineq", "fun": cons}],
                               options={"maxiter": 300, "ftol": 1e-12})
                V3 = uv(res.x)
                if any(float(np.max(K4 @ v)) > HALF + 1e-8 for v in V3):
                    continue
                if any(float(np.max(R @ v)) > cosd + 1e-8 for v in V3):
                    continue
                best = min(best, max(float(V3[a] @ V3[b])
                                     for a, b in combinations(range(3), 2)))
            shown = f"{best:.9f}" if best < 1.5 else "none feasible"
            print(f"    {str(c):<15} {delta:>7.1f} {shown:>38}", flush=True)
        print()
    print("    at margin zero the minimiser is three of the removed roots;")
    print("    a positive margin pushes the value above 1/2, so the doorways")
    print("    at twenty roots hold two further directions and not three.")
    print("    This part is a search over local minima and is reported as a")
    print("    computation, not as a proof.")

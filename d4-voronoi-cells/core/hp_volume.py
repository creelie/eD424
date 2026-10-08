#!/usr/bin/env python3
"""
High-precision (mpmath) polytope volume, to get past double-precision
qhull's noise floor.

VOLUME METHOD (star triangulation from the origin, standard for a
polytope containing the origin in its interior):
  1. Enumerate vertices exactly/high-precision: every size-4 subset of
     the n facet normals gives a candidate vertex by solving <n_k,x>=1
     for k in the subset (4x4 linear solve); keep it if it satisfies
     <n_k,x> <= 1 (+tol) for ALL n facets.
  2. For each facet k, collect its vertex set (>=4 points, lying in a
     3D affine hyperplane). Project to 3D local coordinates (double
     precision is fine here -- only used to FIND the triangulation
     combinatorics, not for any volume number), take the 3D convex
     hull's surface triangles, then star-triangulate the solid facet
     from one of its own vertices into tetrahedra (4 points each).
  3. Each tetrahedron, together with the origin, forms a 4-simplex;
     its 4-volume is (1/4!)|det(M)| where M's rows are the tetrahedron's
     4 actual high-precision vertex vectors (origin contributes a zero
     row after the standard vertex-difference reduction, since it IS
     one of the 5 simplex vertices). Sum over all tetrahedra, all facets.
"""
import mpmath as mp
import itertools
import numpy as np
from scipy.spatial import ConvexHull


def find_vertices(dirs_mp, tol):
    n = len(dirs_mp)
    verts = []
    vert_facets = []
    for combo in itertools.combinations(range(n), 4):
        A4 = mp.matrix([[dirs_mp[k][i] for i in range(4)] for k in combo])
        try:
            detA = mp.det(A4)
        except Exception:
            continue
        if abs(detA) < mp.mpf('1e-6'):
            continue
        b = mp.matrix([1, 1, 1, 1])
        try:
            x = mp.lu_solve(A4, b)
        except Exception:
            continue
        ok = True
        tight = []
        for k in range(n):
            val = sum(dirs_mp[k][i] * x[i] for i in range(4))
            if val > 1 + tol:
                ok = False
                break
            if val > 1 - tol:
                tight.append(k)
        if ok:
            verts.append(x)
            vert_facets.append(frozenset(tight))
    return verts, vert_facets


def tangent_basis_mp(nk):
    std_basis = [mp.matrix([1, 0, 0, 0]), mp.matrix([0, 1, 0, 0]),
                 mp.matrix([0, 0, 1, 0]), mp.matrix([0, 0, 0, 1])]
    nknorm2 = sum(nk[i] ** 2 for i in range(4))
    tb = []
    for e in std_basis:
        proj = sum(e[i] * nk[i] for i in range(4)) / nknorm2
        v = mp.matrix([e[i] - proj * nk[i] for i in range(4)])
        for t in tb:
            d = sum(v[i] * t[i] for i in range(4))
            v = mp.matrix([v[i] - d * t[i] for i in range(4)])
        nrm = mp.sqrt(sum(v[i] ** 2 for i in range(4)))
        if nrm > mp.mpf('1e-8'):
            tb.append(mp.matrix([v[i] / nrm for i in range(4)]))
        if len(tb) == 3:
            break
    return tb


def det4(p1, p2, p3, p4):
    M = mp.matrix([[p1[i] for i in range(4)],
                    [p2[i] for i in range(4)],
                    [p3[i] for i in range(4)],
                    [p4[i] for i in range(4)]])
    try:
        return mp.det(M)
    except Exception:
        return mp.mpf(0)


def hp_volume(dirs_mp, tol=None, prec=50):
    mp.mp.dps = prec
    if tol is None:
        tol = mp.mpf('1e-' + str(prec // 2))
    verts, vert_facets = find_vertices(dirs_mp, tol)
    n = len(dirs_mp)
    facet_verts = {k: [] for k in range(n)}
    for vi, tight in enumerate(vert_facets):
        for k in tight:
            facet_verts[k].append(vi)

    total_vol = mp.mpf(0)
    for k in range(n):
        vids = facet_verts[k]
        if len(vids) < 4:
            continue
        tb = tangent_basis_mp(dirs_mp[k])
        centroid = [mp.mpf(0)] * 4
        for vi in vids:
            v = verts[vi]
            for i in range(4):
                centroid[i] += v[i]
        for i in range(4):
            centroid[i] /= len(vids)
        pts3 = []
        for vi in vids:
            v = verts[vi]
            rel = [v[i] - centroid[i] for i in range(4)]
            c = [float(sum(rel[i] * tb[j][i] for i in range(4))) for j in range(3)]
            pts3.append(c)
        pts3 = np.array(pts3)
        try:
            hull3 = ConvexHull(pts3, qhull_options='QJ')
        except Exception as e:
            raise RuntimeError(f"facet {k} 3D hull failed: {e}")
        # star-triangulate the SOLID facet from local vertex 0
        apex_local = 0
        tetra_count = 0
        for simplex in hull3.simplices:
            if apex_local in simplex:
                continue
            a, b, c = simplex
            p_apex = verts[vids[apex_local]]
            pa, pb, pc = verts[vids[a]], verts[vids[b]], verts[vids[c]]
            vol4 = abs(det4(p_apex, pa, pb, pc)) / mp.mpf(24)
            total_vol += vol4
            tetra_count += 1
    return total_vol


if __name__ == "__main__":
    # sanity check: reference D4 config, all 24 roots -> known volume 8
    roots = []
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0, 0, 0, 0]
                    v[i] = si
                    v[j] = sj
                    roots.append(v)
    mp.mp.dps = 50
    dirs_mp = []
    s2 = mp.sqrt(2)
    for r in roots:
        dirs_mp.append([mp.mpf(x) / s2 for x in r])
    vol = hp_volume(dirs_mp, prec=50)
    print("reference volume (expect 8):", vol)

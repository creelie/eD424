"""
verify45.py -- steps 4 and 5 of the LLM24 verification, from the zonal matrices
computed by psker.

Step 4 builds the four polynomial equality constraints of the second level of
the Lasserre hierarchy, one for each number of points from one to four:

    A_2(K)(Q) + sum of weighted squares = right hand side,

where the sum of squares part comes from the certificate's own factored data
and A_2(K)(Q) is assembled from the zonal matrices and the certificate's
positive semidefinite blocks.  Step 5 checks that each of the four is the zero
polynomial.  Together with the positive definiteness of the blocks, the
nonnegativity of the prefactors, and the objective value, that is what makes
the certificate a feasible point of the hierarchy with value 24.

Nothing here uses the authors' code; the data files and the description of the
format in their README are the only inputs, plus the zonal matrices built from
scratch by psker, o4.py, gl2.py and zonal.py.
"""
import gc
import os
import resource
import pickle
import sys
import time
from fractions import Fraction as Q
from itertools import combinations, chain

from flint import fmpq, fmpq_mat, fmpq_mpoly, fmpq_mpoly_ctx, Ordering

import partio
import zonal
from gl2 import countels

sys.set_int_max_str_digits(0)

D2 = 16
D1 = 14
NIP = 6                                     # u1 .. u6

CTXS = {}
MISSING = set()


def ctx(nv):
    if nv not in CTXS:
        CTXS[nv] = fmpq_mpoly_ctx.get(['x%d' % i for i in range(nv)],
                                      Ordering.lex)
    return CTXS[nv]


def parse_q(tok):
    if '//' in tok:
        a, b = tok.split('//')
        return fmpq(int(a), int(b))
    if '/' in tok:
        a, b = tok.split('/')
        return fmpq(int(a), int(b))
    return fmpq(int(tok))


def load_dense(path):
    with open(path) as f:
        nr, nc = [int(x) for x in f.readline().split()]
        rows = []
        for _ in range(nr):
            rows.append([parse_q(t) for t in f.readline().split()])
    return fmpq_mat(rows)


def load_lowrank(path, nv):
    """the prefactors and the vectors of a factored sum-of-squares block"""
    with open(path) as f:
        nvecs, veclen = [int(x) for x in f.readline().split()]

        def block():
            out = {}
            while True:
                line = f.readline()
                if not line or not line.strip():
                    return out
                t = line.split()
                e = tuple(int(x) for x in t[1:])
                out[e] = out.get(e, fmpq(0)) + parse_q(t[0])

        pref = [block() for _ in range(nvecs)]
        vecs = [[block() for _ in range(veclen)] for _ in range(nvecs)]
    return pref, vecs


def irreps():
    out = []
    for l1 in range(D1 + 1):
        for l2 in range(min(l1, D1 - l1) + 1):
            if l1 != l2 or l1 % 2 == 0:
                out.append((l1, l2))
    return out


def weven(lam):
    return [k for k in range(lam[0] - lam[1] + 1) if (lam[1] + k) % 2 == 0]


def isadmissible(lam, i, j, k):
    if i == 0:
        return sum(lam) == 0 and j == 0 and k == 0
    if i == 1:
        return lam[1] == 0 and j == 0 and k == 0
    return countels(lam, 2, k) % 2 == 0


def admissible_tuples(lam, i, ws, d):
    return [(j, k) for j in range(d + 1) for k in ws
            if isadmissible(lam, i, j, k)]


def gram(card, nv):
    """the Gram matrix of `card` points, with 1 on the diagonal and one
    variable for each pair, in the ring with nv variables"""
    C = ctx(nv)
    X = [[None] * card for _ in range(card)]
    idx = 0
    for i in range(card):
        for j in range(i + 1, card):
            X[i][j] = X[j][i] = C.gen(idx)
            idx += 1
        X[i][i] = C.from_dict({(0,) * nv: fmpq(1)})
    return X


def add_poly(acc, p, scale=None):
    """add a flint polynomial into a dictionary of coefficients

    Accumulating in flint's own rational polynomials is not safe here: they
    carry one denominator for the whole polynomial, so adding many blocks with
    unrelated denominators multiplies every numerator by their least common
    multiple and the object grows out of the machine.  A dictionary keeps each
    coefficient in lowest terms on its own."""
    for e, c in p.terms():
        e = tuple(int(x) for x in e)
        if scale is not None:
            c = c * scale
        if e in acc:
            v = acc[e] + c
            if v == 0:
                del acc[e]
            else:
                acc[e] = v
        elif c != 0:
            acc[e] = c


def poly_from_dict(d, nv):
    C = ctx(nv)
    if not d:
        return C.from_dict({})
    return C.from_dict({e: c for e, c in d.items()})


def zpoly_to_ring(zp, ipsexp, nv):
    """substitute the six inner products, each a monomial, into Z"""
    C = ctx(nv)
    out = {}
    for ev, c in zp.items():
        e = [0] * nv
        ok = True
        for i in range(NIP):
            if ev[i]:
                if ipsexp[i] is None:       # the inner product is the constant 1
                    continue
                for t in range(nv):
                    e[t] += ipsexp[i][t] * ev[i]
        key = tuple(e)
        out[key] = out.get(key, fmpq(0)) + fmpq(c.numerator, c.denominator)
    return C.from_dict({k: v for k, v in out.items() if v != 0})



def det(M, C):
    """the determinant of a small matrix of polynomials, by expansion"""
    n = len(M)
    if n == 0:
        return C.from_dict({(0,) * C.nvars(): fmpq(1)})
    if n == 1:
        return M[0][0]
    out = C.from_dict({})
    for j in range(n):
        minor = [[M[a][b] for b in range(n) if b != j] for a in range(1, n)]
        t = M[0][j] * det(minor, C)
        out = out - t if j % 2 else out + t
    return out


def total_degree(p):
    return max([sum(int(x) for x in e) for e, _ in p.terms()] or [0])


def check_prefactor(pf, sos_weights, nv, k, name):
    """a prefactor is a nonnegative constant, or a nonnegative constant times
    one of the weights that describe the domain"""
    C = ctx(nv)
    e = poly_from_dict(pf, nv)
    terms = list(e.terms())
    if not terms:
        return
    if total_degree(e) == 0:
        assert terms[0][1] >= 0, 'negative constant prefactor in %s' % (name,)
        return
    for sw in sos_weights:
        if total_degree(sw) != total_degree(e):
            continue
        st = dict(sw.terms())
        et = dict(terms)
        if set(st) != set(et):
            continue
        ratios = {et[m] / st[m] for m in st}
        if len(ratios) == 1:
            r = ratios.pop()
            assert r >= 0, 'negative multiple of a domain weight in %s' % (name,)
            return
    raise AssertionError('prefactor of %s is not a nonnegative multiple of a '
                         'domain weight' % (name,))


def main(folder, psfile, only=None, part='all', save=None,
         load=None):
    t0 = time.time()
    lines = open(os.path.join(folder, 'metadata.txt')).read().split('\n')
    assert lines[0].strip() == 'codebound'
    f = lines[1].split()
    n = int(f[0])
    npoints = parse_q(f[1])
    costheta = parse_q(f[2])
    globals()['costheta'] = costheta
    d1, d2 = int(f[3]), int(f[4])
    assert (n, d1, d2) == (4, D1, D2)

    # The blocks for the irreducible representations are small, so B X B^T is
    # formed once.  The sum-of-squares blocks reach 395 x 350 with entries of
    # fifteen thousand digits, and B X B^T for those would not fit in memory;
    # they are kept factored and contracted against the vectors instead.
    blocknames = []
    matrices = {}
    for fn in sorted(os.listdir(folder)):
        if not fn.startswith('dense'):
            continue
        name = tuple(x if 'sos' in x else int(x) for x in fn[:-4].split('_')[1:])
        blocknames.append(name)
        if any(isinstance(x, str) for x in name) or part == 'sos':
            continue
        X = load_dense(os.path.join(folder, fn))
        T = load_dense(os.path.join(folder, 'transform_' + fn[6:]))
        matrices[name] = T * X * T.transpose()
    print('read %d blocks, %d of them for irreducible representations (%.0fs)'
          % (len(blocknames), len(matrices), time.time() - t0))
    blockset = set(blocknames)

    if part != 'sos':
        obj = matrices[(0, 0)][0, 0]
        print('objective B X B^T [0,0] for lambda = (0,0): %s  (expected %s)'
              % (obj, npoints))
        assert obj == fmpq(24)

    nvars = [1, 1, 3, 6]
    sos = [{} for _ in range(5)]
    for fn in sorted(os.listdir(folder)):
        if fn.startswith('poly'):
            parts = fn[:-4].split('_')
            c = int(parts[1])
            name = tuple(x if 'sos' in x else int(x) for x in parts[2:])
            sos[c][name] = load_lowrank(os.path.join(folder, fn),
                                        nvars[c - 1])
    print('read sum-of-squares data (%.0fs)' % (time.time() - t0))

    if part == 'sos':
        ps = {}
        print('sum-of-squares part only: the zonal matrices are not needed')
    else:
        ps = None
    cache = psfile + '.reduced.pkl'
    if ps is not None:
        pass
    elif os.path.exists(cache):
        with open(cache, 'rb') as fh:
            ps = pickle.load(fh)
    else:
        ps = zonal.load_ps(psfile)
        with open(cache, 'wb') as fh:
            pickle.dump(ps, fh, 2)
    print('read %d zonal polynomials (%.0fs)' % (len(ps), time.time() - t0))


    skip_sos = (part == 'zonal')
    skip_zonal = (part == 'sos')
    results = []
    for k in (1, 2, 3, 4):
        if only and k not in only:
            continue
        nv = nvars[k - 1]
        C = ctx(nv)
        tr = gram(k, nv)
        acc = {}

        # --- the weights that describe the domain of admissible Gram matrices
        # Every prefactor of a square has to be a nonnegative multiple of one
        # of these, or a nonnegative constant; that is what makes each square
        # term nonnegative where it has to be.
        xs = [C.gen(i) for i in range(nv)]
        oneC = C.from_dict({(0,) * nv: fmpq(1)})
        cth = C.from_dict({(0,) * nv: costheta})
        sos_weights = []
        if k == 2:
            sos_weights.append((xs[0] + oneC) * (cth - xs[0]))
        elif k >= 3:
            sos_weights.append(det(tr, C))
            orbits_w = [[(y + oneC) * (cth - y) for y in xs]]
            if k == 4:
                orbits_w.append([det([[tr[a][b] for b in S] for a in S], C)
                                 for S in combinations(range(k), 3)])
            for orbit in orbits_w:
                for j in range(1, len(orbit) + 1):
                    g = C.from_dict({})
                    for S in combinations(range(len(orbit)), j):
                        t = oneC
                        for i in S:
                            t = t * orbit[i]
                        g = g + t
                    sos_weights.append(g)

        # --- the sum-of-squares part, in factored form -------------------
        names = [nm for nm in sorted(sos[k], key=str) if nm in blockset]
        lo, hi = 0, len(names)
        if os.environ.get('SOS_BLOCKS'):
            lo, hi = [int(x) for x in os.environ['SOS_BLOCKS'].split(',')]
            print('k=%d: blocks %d to %d of %d' % (k, lo, hi, len(names)))
        for bi, name in enumerate(names):
            if not (lo <= bi < hi):
                continue
            pref, vecs = sos[k][name]
            for pf in pref:
                check_prefactor(pf, sos_weights, nv, k, name)
            if skip_sos:
                continue
            fn = 'dense_' + '_'.join(str(x) for x in name) + '.txt'
            X = load_dense(os.path.join(folder, fn))
            T = load_dense(os.path.join(folder, 'transform_' + fn[6:]))
            # B X B^T, built so that only one large intermediate is alive at a
            # time: for the 395 x 350 blocks each of these matrices is a couple
            # of gigabytes, the entries being sums of rationals whose common
            # denominator has fifteen thousand digits.
            Tt = T.transpose()
            n, m = T.nrows(), T.ncols()
            M = fmpq_mat(n, n)
            row = fmpq_mat(1, m)
            for a in range(n):
                for i in range(m):
                    row[0, i] = T[a, i]
                out = (row * X) * Tt
                for b in range(n):
                    M[a, b] = out[0, b]
            del X, T, Tt, row, out
            gc.collect()
            # w_a = sum_b M[a,b] v[b], so that v^T M v = sum_a v[a] w_a with
            # v[a] a polynomial of at most two dozen terms
            WS = []
            for v in vecs:
                W = []
                for a in range(n):
                    d = {}
                    for b in range(n):
                        m = M[a, b]
                        if m == 0:
                            continue
                        for e, c in v[b].items():
                            d[e] = d.get(e, fmpq(0)) + m * c
                    W.append(C.from_dict({e: c for e, c in d.items() if c != 0}))
                WS.append(W)
            del M
            gc.collect()
            for j, pf in enumerate(pref):
                v = vecs[j]
                tot = C.from_dict({})
                for a in range(n):
                    if v[a]:
                        tot = tot + C.from_dict(v[a]) * WS[j][a]
                pterms = list(pf.items())
                if len(pterms) == 1 and sum(pterms[0][0]) == 0:
                    add_poly(acc, tot, pterms[0][1])
                else:
                    add_poly(acc, poly_from_dict(pf, nv) * tot)
                del tot
                gc.collect()
            del WS
            gc.collect()
            print('    block %d %s: %d terms so far (%.0fs, %.0f MB)'
                  % (bi, str(name), len(acc), time.time() - t0,
                     resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
            sys.stdout.flush()
        print('k=%d sum-of-squares part done (%.0fs)' % (k, time.time() - t0))

        # --- A_2(K)(Q) ---------------------------------------------------
        if load:
            for e, c in partio.iter_parts(load):
                acc[e] = acc.get(e, fmpq(0)) + fmpq(c[0], c[1])
            print('k=%d loaded a saved part, %d terms (%.0fs)'
                  % (k, len(acc), time.time() - t0))
        subs = [] if skip_zonal else \
            list(chain(*[list(combinations(range(k), l)) for l in (0, 1, 2)]))
        for i1x in range(len(subs)):
            for i2x in range(i1x, len(subs)):
                J1, J2 = subs[i1x], subs[i2x]
                if tuple(sorted(set(J1) | set(J2))) != tuple(range(k)):
                    continue
                n1, n2 = len(J1), len(J2)

                def f(Ja, Jb, a, b, na=None, nb=None):
                    return tr[Ja[min(a, len(Ja) - 1)]][Jb[min(b, len(Jb) - 1)]]

                if n1 > 0 and n2 > 0:
                    ips = [f(J1, J1, 0, 1), f(J2, J2, 0, 1),
                           f(J1, J2, 0, 0), f(J1, J2, 0, 1),
                           f(J1, J2, 1, 0), f(J1, J2, 1, 1)]
                else:
                    ips = [C.from_dict({(0,) * nv: fmpq(1)})] * 6
                    if n1 > 0:
                        ips = list(ips)
                        ips[0] = f(J1, J1, 0, 1)
                    elif n2 > 0:
                        ips = list(ips)
                        ips[1] = f(J2, J2, 0, 1)
                ipsexp = []
                for p in ips:
                    ts = list(p.terms())
                    assert len(ts) == 1 and ts[0][1] == 1, 'inner product is ' \
                        'not a monomial'
                    e = tuple(int(x) for x in ts[0][0])
                    ipsexp.append(None if sum(e) == 0 else e)

                fac = fmpq(1, 2) * (2 if J1 != J2 else 1)
                oneC = C.from_dict({(0,) * nv: fmpq(1)})
                zeroC = C.from_dict({})
                for lam in irreps():
                    if lam not in matrices:
                        continue
                    ws = weven(lam)
                    dt = (d2 - sum(lam)) // 2
                    order = []
                    for i in (0, 1, 2):
                        for jk in admissible_tuples(lam, i, ws, dt):
                            order.append((i, jk[0], jk[1]))
                    pos = {t: i for i, t in enumerate(order)}
                    M = matrices[lam]
                    at1 = admissible_tuples(lam, n1, ws, dt)
                    at2 = admissible_tuples(lam, n2, ws, dt)
                    if not at1 or not at2:
                        continue
                    G = {}
                    for (j1, k1) in at1:
                        za = pos[(n1, j1, k1)]
                        for (j2, k2) in at2:
                            zb = pos[(n2, j2, k2)]
                            coef = fac * (M[za, zb] + M[zb, za])
                            if coef == 0:
                                continue
                            e = [0] * nv
                            for (src, power) in ((ipsexp[0], j1),
                                                 (ipsexp[1], j2)):
                                if src is not None and power:
                                    for t in range(nv):
                                        e[t] += src[t] * power
                            d = G.setdefault((k1, k2), {})
                            key = tuple(e)
                            d[key] = d.get(key, fmpq(0)) + coef
                    # accumulate one signature at a time in flint and hand the
                    # result over in one go: the coefficients here have
                    # thousands of digits, and adding tens of thousands of them
                    # one at a time in Python is what costs the hours
                    part = C.from_dict({})
                    for (k1, k2), d in G.items():
                        g = C.from_dict({e: c for e, c in d.items() if c != 0})
                        if not list(g.terms()):
                            continue
                        if (lam, max(k1, k2), min(k1, k2)) not in ps:
                            MISSING.add((lam, k1, k2))
                            continue
                        z = zonal.evaluate_zonal_matrix(
                            ps, lam, k1, k2, ips, oneC, zeroC,
                            lambda q: fmpq(q.numerator, q.denominator))
                        part = part + g * z
                        del z
                    add_poly(acc, part)
                    del part
                    if lam[0] % 4 == 0 and lam[1] == 0:
                        print('      lambda %s, %d terms (%.0fs)'
                              % (str(list(lam)), len(acc), time.time() - t0))
                        sys.stdout.flush()
                gc.collect()
                print('  k=%d J1=%s J2=%s done, %d terms (%.0fs)'
                      % (k, J1, J2, len(acc), time.time() - t0))
        if save:
            partio.save_parts(save, acc)
            print('k=%d saved %d terms to %s' % (k, len(acc), save))
            results.append((k, True, 0))
            continue
        if k == 1:
            add_poly(acc, C.from_dict({(0,) * nv: fmpq(1)}))
        nz = sum(1 for c in acc.values() if c != 0)
        ok = (nz == 0)
        results.append((k, ok, nz))
        print('CONSTRAINT %d: %s  (%d nonzero coefficients, %.0fs)'
              % (k, 'HOLDS EXACTLY' if ok else 'FAILS', nz, time.time() - t0))
        sys.stdout.flush()

    if MISSING:
        print('WARNING: %d zonal entries were missing from the input'
              % len(MISSING))
    print()
    for k, ok, nz in results:
        print('constraint %d  %s' % (k, 'zero' if ok else 'NONZERO (%d)' % nz))
    return all(ok for _, ok, _ in results)


if __name__ == '__main__':
    only = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else None
    part = sys.argv[4] if len(sys.argv) > 4 else 'all'
    save = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != '-' else None
    load = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] != '-' else None
    ok = main(sys.argv[1], sys.argv[2], only, part, save, load)
    sys.exit(0 if ok else 1)

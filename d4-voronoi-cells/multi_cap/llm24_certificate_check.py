#!/usr/bin/env python3
"""
llm24_certificate_check.py -- an independent re-verification, in exact
rational arithmetic, of the parts of the certificate of

    D. de Laat, N. M. Leijenhorst, W. H. H. de Muinck Keizer,
    Optimality and uniqueness of the D4 root system, arXiv:2404.18794,
    data: 4TU.ResearchData, doi:10.4121/74ce1c25-6fca-4680-8a36-e9c18e7e9594
    (LasserreSphericalCodes.zip, md5 02acd5270f7b3fa799abdeb5291706fd)

that can be checked from the published data alone.  The certificate is
an exact rational feasible point of the second level of the Lasserre
hierarchy for spherical codes in S^3 with pairwise inner product at most
1/2, with objective value 24; it is what Lemma 5.1 of that paper and
thm:m24 of ours rest on.

Their verification procedure (README.txt of the data set) has seven
steps.  This script re-implements, from the data files and from the
description of the format, with none of their code:

  step 1  reading the data and checking its format: the 127 dense blocks
          X_i (60 for the irreducible representations lambda of O(4) with
          |lambda| <= 14, 2 + 15 + 50 for the sum-of-squares multipliers
          of the 2-, 3- and 4-point constraints), the 127 transformation
          matrices B_i, and the 76 files of sum-of-squares prefactors
          and polynomial vectors;
  step 2  positive definiteness of every X_i, by a Cholesky
          factorisation in ball arithmetic (flint's arb through
          python-flint), every pivot being a ball contained in the
          positive reals, with the precision doubled on failure exactly
          as in the original verification; the blocks of size at most
          16 are also checked by an exact LDL^T over the rationals
          (an exact elimination of the large blocks, whose entries have
          up to fifteen thousand digits, exceeds the memory available);
  step 4  the prefactor structure: every prefactor of a sum-of-squares
          term is a nonnegative constant or a nonnegative multiple of one
          of the weights describing the domain, (u+1)(1/2-u) and its
          elementary symmetric functions, and the principal minors of the
          Gram matrix; this is what makes every sum-of-squares term
          nonnegative on the domain of admissible Gram matrices;
  step 6  the objective: the (1,1) entry of B X B^T for lambda = (0,0)
          is exactly 24;
  step 7  the inner products: the two-point sum-of-squares polynomial
          p_2(u), built exactly from the data, vanishes at -1, -1/2, 0
          and 1/2 and, by a Sturm sequence computed over the rationals,
          nowhere else on [-1, 1/2].

Steps 3 and 5, the construction of the zonal matrices Z_lambda as
polynomials in the six inner products of four points and the check that
A_2 K(Q) + (sum of squares) = rhs holds as a polynomial identity for
|Q| = 1, 2, 3, 4, are not part of this script: the construction of the
zonal matrices needs about three days and 128 GB of memory in the authors'
implementation, and the programs of ../zonal do both steps on an ordinary
machine in exact rational arithmetic.  What this script establishes is: the
data are well formed, every block is positive definite, every sum-of-squares
term is nonnegative on the domain, the objective is 24, and with the
polynomial identities of step 5 (checked in ../zonal)
every 24-point code of minimal angle 60 degrees has all its inner products
in {-1, -1/2, 0, 1/2}.  The identities themselves are checked by the
programs of ../zonal (steps 3 and 5), in exact rational arithmetic.

Also checked, as a test of the reading of the data: the block for each
lambda has the number of rows the paper's description prescribes (the
number of admissible tuples (i, j, k), Section 2.3 of theirs).

Usage: python3 llm24_certificate_check.py /path/to/LasserreSphericalCodes/proofs/4_24
(the data set is redistributed in ../third_party/llm24-certificate, or
download LasserreSphericalCodes.zip from the DOI above and unpack it).  Writes llm24_out/llm24_p2.txt, the exact
coefficients of p_2, from which lean/gen_innerproducts_lean.py generates
lean/D4InnerProducts.lean.  Needs python-flint (pip install python-flint).
Exits nonzero if any check fails.  Runtime about half an hour on two cores.
"""

import os, sys, time, itertools
from flint import fmpq, fmpq_mat, fmpq_poly, fmpq_mpoly_ctx, Ordering, arb, ctx

sys.set_int_max_str_digits(0)      # the rational entries have up to tens of thousands of digits

RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print("[%s] %s" % ("PASS" if ok else "FAIL", name), flush=True)
    if detail:
        for line in detail.splitlines():
            print("       " + line)


def parse_q(s):
    if "//" in s:
        a, b = s.split("//")
        return fmpq(int(a), int(b))
    return fmpq(int(s))


def load_dense(fname):
    with open(fname) as f:
        r, c = (int(x) for x in f.readline().split())
        assert r > 0 and c > 0
        rows = []
        for line in f:
            if line.strip() == "":
                continue
            rows.append([parse_q(x) for x in line.split()])
    assert len(rows) == r and all(len(x) == c for x in rows), fname
    return fmpq_mat(rows)


def load_low_rank(fname, ctx):
    """prefactors (list of polynomials) and vectors (list of lists of polynomials)."""
    with open(fname) as f:
        nvecs, veclength = (int(x) for x in f.readline().split())
        assert nvecs > 0 and veclength > 0

        def read_poly():
            d = {}
            while True:
                line = f.readline()
                if line == "" or line.strip() == "":
                    break
                parts = line.split()
                c = parse_q(parts[0])
                ev = tuple(int(x) for x in parts[1:])
                d[ev] = d.get(ev, fmpq(0)) + c
            return ctx.from_dict(d) if d else ctx.from_dict({})
        pref = [read_poly() for _ in range(nvecs)]
        vecs = [[read_poly() for _ in range(veclength)] for _ in range(nvecs)]
    return pref, vecs


def blockname(fname, prefix):
    parts = fname[len(prefix):-4].split("_")
    return tuple(p if p.startswith("sos") else int(p) for p in parts)


def ldlt_positive(M):
    """Exact LDL^T over the rationals (used for the small blocks only);
    returns (ok, least pivot)."""
    n = M.nrows()
    A = [[M[i, j] for j in range(n)] for i in range(n)]
    least = None
    for k in range(n):
        p = A[k][k]
        if p <= 0:
            return False, p
        least = p if least is None or p < least else least
        inv = 1 / p
        row = A[k]
        for i in range(k + 1, n):
            f = row[i] * inv
            if f == 0:
                continue
            Ai = A[i]
            for j in range(i, n):
                Ai[j] -= f * row[j]
    return True, least


def cholesky_positive(M, prec=256, maxprec=2 ** 14):
    """Rigorous Cholesky in ball arithmetic (flint arb), as in the original
    verification: every pivot must be a ball lying entirely in (0, inf).
    The precision is doubled until the factorisation goes through or
    maxprec is exceeded.  Returns (ok, precision used, least pivot)."""
    n = M.nrows()
    while prec <= maxprec:
        ctx.prec = prec
        A = [[arb(M[i, j]) for j in range(n)] for i in range(n)]
        L = [[None] * n for _ in range(n)]
        ok = True
        least = None
        for k in range(n):
            d = A[k][k]
            Lk = L[k]
            for j in range(k):
                d -= Lk[j] * Lk[j]
            if not (d > 0):
                ok = False
                break
            least = d if least is None or (d < least) else least
            Lkk = d.sqrt()
            Lk[k] = Lkk
            for i in range(k + 1, n):
                sacc = A[i][k]
                Li = L[i]
                for j in range(k):
                    sacc -= Li[j] * Lk[j]
                Li[k] = sacc / Lkk
        if ok:
            return True, prec, least
        prec *= 2
    return False, prec, None


# ---------------------------------------------------------------- admissible tuples (Section 2.3)
def lambdas(t, d, lmax=None):
    if lmax is None:
        lmax = d
    if d < 0:
        return []
    if d == 0 or t == 0:
        return [[0] * t]
    out = []
    for l1 in range(0, min(d, lmax) + 1):
        for aux in lambdas(t - 1, d - l1, l1):
            out.append([l1] + aux)
    res = []
    for aux in out:
        aux = list(aux)
        while len(aux) > t:
            assert aux[-1] == 0
            aux.pop()
        while len(aux) < t:
            aux.append(0)
        res.append(aux)
    return res


def countels(lam, j, k):
    return lam[0] - k if j == 1 else lam[1] + k


def isadmissible(lam, i, j, k):
    if i == 0:
        return sum(lam) == 0 and j == 0 and k == 0
    if i == 1:
        return lam[1] == 0 and j == 0 and k == 0
    return countels(lam, 2, k) % 2 == 0


def admissible_count(lam, d1, d2):
    ws = [k for k in range(0, lam[0] - lam[1] + 1) if countels(lam, 2, k) % 2 == 0]
    dt = (d2 - sum(lam)) // 2
    return sum(1 for i in range(3) for j in range(dt + 1) for k in ws if isadmissible(lam, i, j, k))


# ---------------------------------------------------------------- weights of the domain
def gram_vars(k, ctx):
    """Symbolic Gram matrix of k unit vectors in the variables of ctx, ordered (1,2),(1,3),...."""
    g = ctx.gens()
    X = [[None] * k for _ in range(k)]
    idx = 0
    for i in range(k):
        X[i][i] = ctx.from_dict({(0,) * len(g): fmpq(1)})
        for j in range(i + 1, k):
            X[i][j] = X[j][i] = g[idx]
            idx += 1
    return X


def det(M):
    n = len(M)
    if n == 1:
        return M[0][0]
    res = None
    for j in range(n):
        minor = [row[:j] + row[j + 1:] for row in M[1:]]
        term = M[0][j] * det(minor)
        if j % 2:
            term = -term
        res = term if res is None else res + term
    return res


def elementary(polys, ctx):
    out = []
    for r in range(1, len(polys) + 1):
        s = None
        for S in itertools.combinations(polys, r):
            p = S[0]
            for q in S[1:]:
                p = p * q
            s = p if s is None else s + p
        out.append(s)
    return out


def domain_weights(k, ctx, costheta):
    g = ctx.gens()
    one = ctx.from_dict({(0,) * len(g): fmpq(1)})
    ws = []
    if k == 2:
        ws.append((g[0] + one) * (one * costheta - g[0]))
    elif k >= 3:
        X = gram_vars(k, ctx)
        ws.append(det(X))
        orbits = [[(y + one) * (one * costheta - y) for y in g]]
        if k == 4:
            orbits.append([det([[X[a][b] for b in S] for a in S]) for S in itertools.combinations(range(4), 3)])
        for orbit in orbits:
            ws.extend(elementary(orbit, ctx))
    return ws


def is_nonneg_multiple(e, weights):
    """e is a nonnegative constant, or c*w with c >= 0 and w a weight."""
    if e.is_zero():
        return True
    if e.total_degree() == 0:
        return e.coeffs()[0] >= 0
    for w in weights:
        if w.total_degree() != e.total_degree():
            continue
        # e = c * w  <=>  every coefficient ratio equal and nonnegative
        ew = dict(zip(e.monoms(), e.coeffs()))
        ww = dict(zip(w.monoms(), w.coeffs()))
        if set(ew) != set(ww):
            continue
        ratios = {ew[m] / ww[m] for m in ew}
        if len(ratios) == 1:
            c = ratios.pop()
            return c >= 0
    return False


# ---------------------------------------------------------------- Sturm sequences
def sturm_count(p, a, b):
    """Number of distinct zeros of the fmpq_poly p in (a, b], p(a) != 0 required."""
    seq = [p, p.derivative()]
    while True:
        r = seq[-2] % seq[-1]
        if r == 0:
            break
        seq.append(-r)
    def var(x):
        signs = [1 if q(x) > 0 else -1 for q in seq if q(x) != 0]
        return sum(1 for i in range(len(signs) - 1) if signs[i] != signs[i + 1])
    return var(a) - var(b)


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "LasserreSphericalCodes/proofs/4_24"
    t0 = time.time()

    # ---- step 1: read ---------------------------------------------------------
    lines = open(os.path.join(folder, "metadata.txt")).read().splitlines()
    assert lines[0] == "codebound"
    n, N, costheta, d1, d2, delta = lines[1].split()
    n, d1, d2, delta = int(n), int(d1), int(d2), int(delta)
    N = parse_q(N); costheta = parse_q(costheta)
    innerproducts = [parse_q(x) for x in lines[2].split()]
    record("metadata: n = 4, N = 24, cos theta = 1/2, d1 = 14, d2 = 16, inner products -1, -1/2, 0, 1/2",
           n == 4 and N == 24 and costheta == fmpq(1, 2) and d1 == 14 and d2 == 16 and delta == 16
           and innerproducts == [fmpq(-1), fmpq(-1, 2), fmpq(0), fmpq(1, 2)],
           "metadata.txt: " + " | ".join(lines))

    files = sorted(os.listdir(folder))
    matrices, transforms = {}, {}
    for fn in files:
        if fn.startswith("dense"):
            b = blockname(fn, "dense_")
            matrices[b] = load_dense(os.path.join(folder, fn))
            tf = "transform_" + fn[6:]
            assert tf in files, tf
            transforms[b] = load_dense(os.path.join(folder, tf))
    ok = True
    for b, X in matrices.items():
        B = transforms[b]
        ok &= X.nrows() == X.ncols() == B.ncols()
    record("step 1: 127 dense blocks X_i read, each square and matching its transformation B_i",
           len(matrices) == 127 and ok,
           "irreducible representations: %d blocks; sos2: %d; sos3: %d; sos4: %d; total dimension %d, largest %d"
           % (sum(1 for b in matrices if isinstance(b[0], int)),
              sum(1 for b in matrices if b[0] == "sos2"), sum(1 for b in matrices if b[0] == "sos3"),
              sum(1 for b in matrices if b[0] == "sos4"),
              sum(X.nrows() for X in matrices.values()), max(X.nrows() for X in matrices.values())))

    # the irreps and their admissible-tuple counts
    irreps = [lam for lam in lambdas(2, d1) if lam[0] != lam[1] or lam[0] % 2 == 0]
    irrep_blocks = {b for b in matrices if isinstance(b[0], int)}
    expected = {tuple(lam): admissible_count(lam, d1, delta) for lam in irreps}
    ok = all(b in expected for b in irrep_blocks)
    detail = []
    for b in sorted(irrep_blocks):
        rows = transforms[b].nrows()
        if rows != expected[b]:
            ok = False
            detail.append("lambda = %s: transform has %d rows, admissible tuples %d" % (b, rows, expected[b]))
    record("the block of every lambda has as many rows as there are admissible tuples (i, j, k) for that lambda",
           ok, "%d of the %d signatures with |lambda| <= 14 carry a nonzero block%s"
           % (len(irrep_blocks), len(expected), ("; " + "; ".join(detail)) if detail else ""))

    ctxs = {k: fmpq_mpoly_ctx.get(tuple("x%d" % i for i in range(max(1, k * (k - 1) // 2))), Ordering.lex) for k in range(1, 5)}
    sos = {k: {} for k in range(1, 5)}
    for fn in files:
        if fn.startswith("poly"):
            parts = fn[:-4].split("_")
            k = int(parts[1])
            b = tuple(p if p.startswith("sos") else int(p) for p in parts[2:])
            sos[k][b] = load_low_rank(os.path.join(folder, fn), ctxs[k])
    record("step 1: 76 sum-of-squares files read (1 + 2 + 15 + 58 for the 1-, 2-, 3-, 4-point constraints)",
           [len(sos[k]) for k in range(1, 5)] == [1, 2, 15, 58],
           "counts: %s; the blocks without a dense file are zero and are skipped, as in the original"
           % [len(sos[k]) for k in range(1, 5)])

    # ---- step 2: positive definiteness ---------------------------------------
    print("step 2: Cholesky in ball arithmetic of every block (and exact LDL^T of the blocks of size at most 16) ...", flush=True)
    t1 = time.time()
    allpos = True
    precs = {}
    least_over_all = None
    exact_blocks = 0
    for b in sorted(matrices, key=lambda b: matrices[b].nrows()):
        X = matrices[b]
        ts = time.time()
        ok, prec, least = cholesky_positive(X)
        if not ok:
            allpos = False
            print("   block %s: Cholesky failed up to precision %d bits" % (b, prec), flush=True)
        else:
            precs[prec] = precs.get(prec, 0) + 1
            lo = least.lower()
            least_over_all = lo if least_over_all is None or lo < least_over_all else least_over_all
        if X.nrows() <= 16:
            ok2, least2 = ldlt_positive(X)
            exact_blocks += 1
            if not ok2:
                allpos = False
                print("   block %s: exact LDL^T finds a nonpositive pivot %s" % (b, least2), flush=True)
        print("   block %-18s %4d x %-4d  %s  %5.1f s" % (str(b), X.nrows(), X.ncols(), "PD" if ok else "FAILED", time.time() - ts), flush=True)
    record("step 2: every one of the 127 blocks X_i is positive definite (Cholesky in ball arithmetic, every pivot a ball inside (0, inf))",
           allpos, "%.0f s; precision used (bits: number of blocks) %s; least pivot at least %s; the %d blocks of size at most 16 also checked by exact rational LDL^T"
           % (time.time() - t1, sorted(precs.items()), least_over_all, exact_blocks))

    # ---- step 6: objective ----------------------------------------------------
    B = transforms[(0, 0)]; X = matrices[(0, 0)]
    M = B * X * B.transpose()
    record("step 6: the objective K(empty, empty) = (B X B^T)_{11} for lambda = (0,0) equals 24 exactly",
           M[0, 0] == N, "value %s" % M[0, 0])

    # ---- step 4: prefactor structure ----------------------------------------
    ok = True
    counts = {}
    for k in range(1, 5):
        weights = domain_weights(k, ctxs[k], costheta)
        for b, (pref, vecs) in sos[k].items():
            if b not in matrices:
                continue
            for e in pref:
                counts[k] = counts.get(k, 0) + 1
                if not is_nonneg_multiple(e, weights):
                    ok = False
                    print("   constraint %d block %s: prefactor %s is not a nonnegative multiple of a domain weight" % (k, b, e))
    record("step 4: every sum-of-squares prefactor is a nonnegative constant or a nonnegative multiple of a domain weight",
           ok, "prefactors checked per constraint: %s; weights: (u+1)(1/2-u), Gram determinants, their elementary symmetric functions"
           % counts)

    # ---- step 7: the two-point polynomial and its zeros ----------------------
    ctx2 = ctxs[2]
    p2 = None
    for b, (pref, vecs) in sos[2].items():
        if b not in matrices:
            continue
        M = transforms[b] * matrices[b] * transforms[b].transpose()
        m = M.nrows()
        for e, v in zip(pref, vecs):
            assert len(v) == m
            q = None
            for i in range(m):
                for j in range(m):
                    if M[i, j] == 0:
                        continue
                    term = v[i] * v[j] * M[i, j]
                    q = term if q is None else q + term
            term = e * q
            p2 = term if p2 is None else p2 + term
    # to a univariate polynomial
    coeffs = {}
    for mon, c in zip(p2.monoms(), p2.coeffs()):
        coeffs[mon[0]] = c
    deg = max(coeffs)
    P2 = fmpq_poly([coeffs.get(i, fmpq(0)) for i in range(deg + 1)])
    vals = [P2(ip) for ip in innerproducts]
    record("step 7: the two-point polynomial p_2(u) vanishes at u = -1, -1/2, 0, 1/2",
           all(v == 0 for v in vals), "degree %d; leading coefficient %s" % (deg, P2[deg]))
    # multiplicities and the remaining zeros
    Q = P2
    mult = {}
    x = fmpq_poly([0, 1])
    for ip in innerproducts:
        mult[ip] = 0
        while Q(ip) == 0:
            Q = Q // (x - ip)
            mult[ip] += 1
    # zeros of Q on [-1, 1/2]: Q(-1) and Q(1/2) are nonzero now, so count zeros in (-1, 1/2]
    nz = sturm_count(Q, fmpq(-1), costheta)
    record("step 7: after removing those four zeros, p_2 has no further zero on [-1, 1/2] (Sturm sequence over Q)",
           nz == 0 and Q(fmpq(-1)) != 0 and Q(costheta) != 0,
           "multiplicities at -1, -1/2, 0, 1/2: %s; zeros of the quotient on [-1,1/2]: %d"
           % ([mult[ip] for ip in innerproducts], nz))
    # nonnegativity of p_2 on the domain, as a consistency check of the reading
    samples = [fmpq(-1) + fmpq(3, 2) * fmpq(i, 400) for i in range(401)]
    record("p_2 is nonnegative at 401 rational points of [-1, 1/2] (consistency of the reading; it is a weighted sum of squares)",
           all(P2(s) >= 0 for s in samples))
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm24_out")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "llm24_p2.txt"), "w") as f:
        f.write("# p_2(u) = sum_i c_i u^i, exact coefficients, i = 0..%d\n" % deg)
        for i in range(deg + 1):
            f.write("%d %s\n" % (i, P2[i]))

    ok = all(o for _, o in RESULTS)
    print("=" * 62)
    print("%d of %d checks passed (%.0f s)" % (sum(o for _, o in RESULTS), len(RESULTS), time.time() - t0))
    print("Not part of this script: step 3 (the zonal matrices) and step 5 (the identities")
    print("A_2 K(Q) + SOS = rhs for |Q| = 1..4), which the programs of ../zonal carry out.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

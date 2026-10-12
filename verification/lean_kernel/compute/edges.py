"""Line searches along extreme rays: t* = sup{t >= 0 : J + tY in R}."""
from fractions import Fraction as F
from forms import (qform, sym_from_vec, ldl, blocks, values_upto, dim_of)

def add(J, Y, t):
    n = len(J)
    return [[J[i][j] + t * Y[i][j] for j in range(n)] for i in range(n)]

def q_pd(M):
    Q, r, s = blocks(M)
    return ldl(Q) is not None

def below4(J, Y, t):
    """For M = J + tY with Q(M) pos. def.: list of (k, crossing) for k in M* with M[k] <= 4 and Y[k] < 0,
    crossing = (J[k]-4)/(-Y[k]); also returns whether min M >= 4."""
    M = add(J, Y, t)
    vals = values_upto(M, F(4))
    out = []
    ok = True
    for k, v in vals.items():
        if v < 4:
            ok = False
        yk = qform(Y, k)
        if yk < 0:
            out.append((k, (qform(J, k) - 4) / (-yk)))
    return ok, out

def find_upper(J, Y, maxB=F(1024)):
    """Some k with Y[k] < 0 among k with J[k] <= B, B doubling. Returns t_up or None."""
    B = F(8)
    while B <= maxB:
        vals = values_upto(J, B)
        cands = []
        for k, v in vals.items():
            yk = qform(Y, k)
            if yk < 0:
                cands.append((v - 4) / (-yk))
        if cands:
            return min(cands)
        B *= 2
    return None

def tstar(J, Y, witness=None):
    """Exact t*(J, Y) = max{t : J + tY in R} for a vertex J and a bounded extreme ray Y.
    witness: some k in M* with Y[k] < 0 (gives the first upper bound)."""
    if witness is None:
        t = find_upper(J, Y)
        if t is None:
            return None
    else:
        t = (qform(J, witness) - 4) / (-qform(Y, witness))
    lo = F(0)          # J + lo*Y in R
    hi = None          # J + hi*Y has Q not pos. def. (if not None)
    while True:
        if not q_pd(add(J, Y, t)):
            hi = t
            t = (lo + t) / 2
            continue
        ok, cr = below4(J, Y, t)
        if ok:
            if any(c == t for _, c in cr):
                return t
            lo = t
            if hi is None:
                raise RuntimeError("in R at t with no crossing and no upper bound")
            t = (lo + hi) / 2
            continue
        tc = min(c for _, c in cr)
        assert lo <= tc < t
        t = tc

# check_candidates.jl -- independent exact check of the candidate list of
# Proposition 5.4 from ../data/vertices.txt and ../data/edges.txt.
#
# Candidates are (a) vertex representatives J with det J <= 0, and (b) points
# J + tX with det(J + tX) = 0 and 0 < t < t* on edges (t* = Inf for an
# unbounded edge).  For each candidate the sign of det Q - 256 is decided in
# exact rational arithmetic with Sturm sequences, and every candidate with
# det Q = 256 is checked to be a representation of D4 (rank four, det Q = 256,
# 2 Q^{-1} r integral).
#
# Usage: julia check_candidates.jl ../data

const R = Rational{BigInt}

# ------------------------------------------------------------ integer matrices
function bareiss_det(A::Matrix{BigInt})
    A = copy(A); n = size(A, 1); prev = big(1); sgn = 1
    for c in 1:n
        p = findfirst(i -> A[i, c] != 0, c:n)
        p === nothing && return big(0)
        p = p + c - 1
        if p != c
            A[[c, p], :] = A[[p, c], :]; sgn = -sgn
        end
        for i in c+1:n, j in c+1:n
            A[i, j] = div(A[c, c] * A[i, j] - A[i, c] * A[c, j], prev)
        end
        A[c+1:n, c] .= 0
        prev = A[c, c]
    end
    return sgn * A[n, n]
end

ratdet(A::Matrix{R}) = begin
    den = lcm(map(x -> denominator(x), A))
    bareiss_det(BigInt.(A .* den)) // den^size(A, 1)
end

# ------------------------------------------------------------ polynomials (low degree first)
trim(p) = (while length(p) > 1 && p[end] == 0; pop!(p); end; p)
deg(p) = (q = trim(copy(p)); length(q) == 1 && q[1] == 0 ? -1 : length(q) - 1)
peval(p, x) = foldr((c, acc) -> c + x * acc, p; init = zero(R))
deriv(p) = length(p) <= 1 ? R[0] : R[(i - 1) * p[i] for i in 2:length(p)]

function pdivrem(a, b)
    a = trim(copy(a)); b = trim(copy(b))
    db = length(b) - 1
    q = zeros(R, max(length(a) - db, 1))
    while deg(a) >= db && deg(a) >= 0
        k = length(a) - 1 - db
        c = a[end] // b[end]
        q[k+1] = c
        for i in 1:length(b)
            a[k+i] -= c * b[i]
        end
        trim(a)
        if length(a) == 1 && a[1] == 0
            break
        end
    end
    return trim(q), trim(a)
end

function pgcd(a, b)
    a = trim(copy(a)); b = trim(copy(b))
    while deg(b) >= 0
        _, r = pdivrem(a, b)
        a, b = b, r
    end
    return a .// a[end]
end

function sturm(p)
    s = [trim(copy(p)), trim(deriv(p))]
    while deg(s[end]) > 0
        _, r = pdivrem(s[end-1], s[end])
        deg(r) < 0 && break
        push!(s, -r)
    end
    return s
end

function changes(s, x)
    v = [sign(peval(q, x)) for q in s]
    v = filter(!=(0), v)
    count(i -> v[i] != v[i+1], 1:length(v)-1)
end

# distinct roots of p in (a, b], p square-free, via Sturm
nroots(s, a, b) = changes(s, a) - changes(s, b)

function interpolate(xs, ys)
    # Newton form, then expand to monomial coefficients
    n = length(xs); c = copy(ys)
    for j in 2:n, i in n:-1:j
        c[i] = (c[i] - c[i-1]) // (xs[i] - xs[i-j+1])
    end
    p = R[c[n]]
    for i in n-1:-1:1
        # p = p * (t - xs[i]) + c[i]
        q = zeros(R, length(p) + 1)
        for k in 1:length(p)
            q[k+1] += p[k]; q[k] -= xs[i] * p[k]
        end
        q[1] += c[i]; p = q
    end
    return trim(p)
end

# isolate the roots of square-free p in (lo, hi): list of (a, b) with exactly one root in (a, b], or (r, r) exact
function isolate(p, lo::R, hi::R)
    s = sturm(p); out = Tuple{R,R}[]
    stack = [(lo, hi)]
    while !isempty(stack)
        a, b = pop!(stack)
        n = nroots(s, a, b)
        n == 0 && continue
        if peval(p, b) == 0 && n == 1
            b < hi && push!(out, (b, b))   # a root at hi is the far end point, a vertex
            continue
        end
        if n == 1
            push!(out, (a, b)); continue
        end
        m = (a + b) / 2
        push!(stack, (a, m)); push!(stack, (m, b))
    end
    return out
end

# sign of h at the unique root of square-free p in (a, b] (or at r if a == b == r)
function sign_at_root(p, h, a::R, b::R)
    a == b && return sign(peval(h, a))
    g = pgcd(p, h)
    if deg(g) > 0
        sg = sturm(g)
        nroots(sg, a, b) > 0 && return 0
    end
    sh = sturm(trim(copy(h)))
    sp = sturm(p)
    while nroots(sh, a, b) > 0 || peval(h, b) == 0
        m = (a + b) / 2
        if peval(p, m) == 0
            return sign(peval(h, m))
        end
        if nroots(sp, a, m) > 0
            b = m
        else
            a = m
        end
    end
    return sign(peval(h, b))
end

# ------------------------------------------------------------ data
function readints(path)
    [parse.(BigInt, split(l)) for l in eachline(path) if !isempty(strip(l))]
end

tomat(v) = R.(permutedims(reshape(v, 5, 5)))

function is_d4(J::Matrix{R})
    Q = J[1:4, 1:4]; r = J[1:4, 5]
    ratdet(J) == 0 || return false
    ratdet(Q) == 256 || return false
    q = Q \ r
    all(x -> denominator(2x) == 1, q)
end

function main(dir)
    V = readints(joinpath(dir, "vertices.txt"))
    E = readints(joinpath(dir, "edges.txt"))
    VJ = [tomat(v[2:26]) for v in V]
    cands = Any[]
    for (i, J) in enumerate(VJ)
        dJ = ratdet(J)
        if dJ <= 0
            g = ratdet(J[1:4, 1:4])
            push!(cands, ("vertex $(i-1)", Float64(g), sign(g - 256), g == 256 && is_d4(J)))
        end
    end
    for e in E
        ei, vi = Int(e[1]), Int(e[2]) + 1
        X = tomat(e[3:27]); t = e[28]; to = e[29]
        hi = to >= 0 ? R(t) : nothing
        J = VJ[vi]
        xs = R.(0:5)
        p = interpolate(xs, [ratdet(J + x * X) for x in xs])
        gq = interpolate(xs, [ratdet((J+x*X)[1:4, 1:4]) for x in xs])
        h = copy(gq); h[1] -= 256
        if deg(p) < 0
            error("edge $ei lies in R_0; not expected")
        end
        deg(p) == 0 && continue
        psf, _ = pdivrem(p, pgcd(p, deriv(p)))
        if hi === nothing
            B = 1 + maximum(abs.(psf[1:end-1] .// psf[end]))
            hi2 = R(ceil(BigInt, B))
        else
            hi2 = hi
        end
        # roots in (0, hi2); a root at 0 is the vertex itself, excluded by starting at 0 (Sturm counts (0, b])
        for (a, b) in isolate(psf, R(0), hi2)
            s = sign_at_root(psf, h, a, b)
            # approximate value for the report
            aa, bb = a, b
            while bb - aa > big(1) // big(10)^12
                m = (aa + bb) / 2
                if peval(psf, m) == 0
                    aa = bb = m; break
                end
                if sign(peval(psf, aa)) * sign(peval(psf, m)) <= 0
                    bb = m
                else
                    aa = m
                end
            end
            gval = Float64(peval(gq, (aa + bb) / 2))
            d4 = false
            if s == 0
                g1 = pgcd(psf, h)
                deg(g1) == 1 || error("equality at an irrational point on edge $ei")
                t0 = -g1[1] / g1[2]
                d4 = is_d4(J + t0 * X)
            end
            push!(cands, ("edge $ei from vertex $(vi-1)", gval, s, d4))
        end
    end
    sort!(cands, by = c -> c[2])
    println(length(cands), " candidates")
    for c in cands
        println("  ", rpad(c[1], 26), " det Q ≈ ", round(c[2], digits = 6), c[3] > 0 ? "  > 256" : c[3] == 0 ? "  = 256" : "  < 256", c[4] ? "  (D4)" : "")
    end
    all(c -> c[3] >= 0, cands) || error("a candidate below 256")
    all(c -> c[3] != 0 || c[4], cands) || error("equality at a candidate that is not D4")
    println("every candidate has det Q >= 256, with equality only at representations of D4")
    println("ALL CHECKS PASSED")
end

main(length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "data"))

# Fixed-cardinality version of assemble.jl: the moment side has exactly FC_N points (its subset
# counts are binomial(N, j), j = 1..4) and maximizes the design defect sum_k w_k S_k, where
#   S_k = sum_{i,j} G_k(<x_i, x_j>),  G_k = U_k/(k+1)  (Gegenbauer polynomials of S^3, G_k(1) = 1).
# Dual (certificate) side: minimize K(0,0) + sum_j binomial(N, j) nu_j + N sum_k w_k subject to
#   (A_t K)(S) + sos(S) - nu_{|S|} = -2 g(u_S) [|S| = 2],   g = sum_k w_k G_k,
# so every real code of exactly N points with inner products in [-1, 1/2 + s] has
#   sum_k w_k S_k <= objective value.  The A blocks are those of the chunk files unchanged.
# Weights: env FC_W = "w1,w2,..." (rationals, for k = 1, 2, ...), default the hole-test weights.
# FC_MODE = "eq" fixes all four subset counts (its feasible set has no interior: the count of points
# then has variance 0, a singular block, and the dual diverges); FC_MODE = "eps" (default) fixes only
# N and asks for at most binomial(N, 2) + FC_EPS pairs, a relaxation with interior points that still
# excludes thinned pseudo-codes (thinning a pseudo-code of B > N points down to N points leaves
# binomial(N, 2) + N(B - N)/(2B) pairs, 0.34 more than binomial(24, 2) when B = 24.7).
function gk_big(k, u)
    a, b = one(u), 2u
    k == 0 && return a
    for _ = 1:k-1; a, b = b, 2u * b - a; end
    return b / (k + 1)
end
function assemble_sdp(files, prec)
    N = parse(Int, get(ENV, "FC_N", "24"))
    w = [parse(Rational{BigInt}, t) for t in split(get(ENV, "FC_W", "103//963,191//963,280//963,230//963,159//963"), ",")]
    Ablocks = Dict{Any, Dict{Int, Union{CL.LowRankMat, CL.ArbRefMatrix}}}(); sizes = Dict{Any, Int}()
    typ = Dict{Int, Int}()
    m = 0
    for f in files
        r = deserialize(f)
        k = parse(Int, basename(f)[2:2])
        m = r.m
        for (nm, d) in r.Ablocks
            dd = get!(Ablocks, nm) do; Dict{Int, Union{CL.LowRankMat, CL.ArbRefMatrix}}(); end
            for (i, M) in d; @assert !haskey(dd, i); dd[i] = M; end
        end
        for (nm, sz) in r.sizes; @assert get(sizes, nm, sz) == sz; sizes[nm] = sz; end
        for i in keys(r.cvals); typ[i] = k; end
    end
    for nm in collect(keys(Ablocks)); isempty(Ablocks[nm]) && (delete!(Ablocks, nm); delete!(sizes, nm)); end
    @assert sort(collect(keys(typ))) == collect(1:m) "missing samples: have $(length(typ)) of $m"
    names = collect(keys(Ablocks))
    A = [[begin
            M = Matrix{Dict{Int, Union{CL.LowRankMat, CL.ArbRefMatrix}}}(undef, 1, 1); M[1, 1] = Ablocks[nm]; M
          end for nm in names]]
    Cb = [CL.ArbRefMatrix(sizes[nm], sizes[nm], prec=prec) for nm in names]
    for (l, nm) in enumerate(names)
        nm == [0, 0] && (Cb[l][1, 1] = 1)
    end
    C = CL.BlockDiagonal([CL.BlockDiagonal(Cb)])
    mode = get(ENV, "FC_MODE", "eps")
    c = CL.ArbRefMatrix(m, 1, prec=prec)
    B = CL.ArbRefMatrix(m, mode == "eq" ? 4 : 1, prec=prec)
    npair = 0
    for i = 1:m
        mode == "eq" ? (B[i, typ[i]] = -1) : (typ[i] == 1 && (B[i, 1] = -1))
        if typ[i] == 2
            M = Ablocks[(:sos2, 1)][i]
            u = BigFloat(M.vs[1][2]) / BigFloat(M.vs[1][1])
            c[i, 1] = -2 * sum(BigFloat(w[k]) * gk_big(k, u) for k in eachindex(w))
            npair += 1
        end
    end
    const0 = CL.Arblib.Arb(N * sum(BigFloat.(w)), prec=prec)
    if mode == "eq"
        b = CL.ArbRefMatrix(4, 1, prec=prec)
        for j = 1:4; b[j, 1] = binomial(N, j); end
        println("FC assembly (eq): N = $N, weights = $(w), m = $m rows ($(npair) pair rows)"); flush(stdout)
        return CL.ClusteredLowRankSDP(false, const0, A, [B], [c], C, b, [names], Any[:nu1, :nu2, :nu3, :nu4], [[(false, 1) for _ in names]])
    end
    # pairs <= binomial(N, 2) + eps: a 1x1 block nu2 >= 0 with entry -1 in every pair row and cost P
    P = binomial(N, 2) + parse(Rational{BigInt}, get(ENV, "FC_EPS", "1//20"))
    one1 = CL.ArbRefMatrix(1, 1, prec=prec); one1[1, 1] = 1
    dnu = Dict{Int, Union{CL.LowRankMat, CL.ArbRefMatrix}}()
    for i = 1:m
        typ[i] == 2 && (dnu[i] = CL.LowRankMat([CL.Arblib.Arb(-1, prec=prec)], [one1], [one1]))
    end
    Mn = Matrix{Dict{Int, Union{CL.LowRankMat, CL.ArbRefMatrix}}}(undef, 1, 1); Mn[1, 1] = dnu
    push!(A[1], Mn); push!(names, (:nu2,))
    Cn = CL.ArbRefMatrix(1, 1, prec=prec); Cn[1, 1] = CL.Arblib.Arb(P, prec=prec)
    C = CL.BlockDiagonal([CL.BlockDiagonal([Cb; [Cn]])])
    b = CL.ArbRefMatrix(1, 1, prec=prec); b[1, 1] = N
    println("FC assembly (eps): N = $N, pairs <= $(Float64(P)), weights = $(w), m = $m rows ($(npair) pair rows)"); flush(stdout)
    return CL.ClusteredLowRankSDP(false, const0, A, [B], [c], C, b, [names], Any[:nu1], [[(false, 1) for _ in names]])
end

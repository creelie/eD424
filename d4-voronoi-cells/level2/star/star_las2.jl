# star_las2.jl -- the second level of the Lasserre hierarchy of de Laat, Leijenhorst
# and de Muinck Keizer (arXiv:2404.18794), changed so that a code may contain pairs
# with inner product up to BL > B as long as these long pairs form a star.
#
# A finite set C of S^3 is admissible when every pair has inner product at most BL and
# the pairs above B all share one point.  An admissible set of 25 points is a code of
# 24 points with inner products at most B together with one further point at inner
# product at most BL with each of them.  With B = 1/2 + 1/125 (24 centres within
# 2.0161 of c) and BL = 0.6141 (a further centre within sqrt 6), a bound below 25 says
# that 24 centres within 2.0161 leave no room for a further centre within sqrt 6.
#
# The admissible sets are those of the hypergraph whose forbidden sets are the pairs
# above BL, the triangles of long pairs and the 4-sets with two disjoint long pairs,
# so the hierarchy applies unchanged; only the domains of the sum-of-squares parts
# change.  The function A_2K on k-sets is symmetric, so it suffices to certify it on
# the admissible k-sets whose long pairs lie in the star at point 1:
#   k = 2: u in [-1, BL];
#   k = 3: u12, u13 in [-1, BL], u23 in [-1, B];
#   k = 4: u12, u13, u14 in [-1, BL], u23, u24, u34 in [-1, B].
# These pieces have less symmetry than the full domain: the sum-of-squares parts use
# monomial bases, summed over the stabiliser of point 1 (SYM=1, the default; SYM=0
# uses no symmetry and samples unisolvent for all polynomials of the degree).  With BL = B the programme is the plain one,
# written less economically, which is the control.
#
# usage: julia --project=$LSC star_las2.jl D1 DELTA PREC B BL [OUT]
#   B, BL rationals (127//250, 6141//10000).  Floating point; exploration only.
using LasserreSphericalCodes, ClusteredLowRankSolver, Nemo, IterTools, Random
const L = LasserreSphericalCodes

Random.seed!(1)

function monomials_upto(x, m)
    R = parent(x[1])
    out = [R(1)]
    m == 0 && return out
    nv = length(x)
    for deg = 1:m
        for c in IterTools.subsets(1:(nv + deg - 1), nv - 1)
            # stars and bars: exponents from the gaps
            e = Int[]; prev = 0
            for ci in c
                push!(e, ci - prev - 1); prev = ci
            end
            push!(e, nv + deg - 1 - prev)
            push!(out, prod(x[i]^e[i] for i = 1:nv))
        end
    end
    out
end

function grid_samples(nv, d, FF, n = binomial(nv + d, nv))
    ch = [ClusteredLowRankSolver.sample_points_chebyshev(2d + 2k, -1, 1) for k = 1:nv]
    ch = [[FF(floor(Int, x * 10^4) // 10^4) for x in v] for v in ch]
    idxs = Set{Vector{Int}}()
    while length(idxs) < n
        push!(idxs, [rand(1:(2d + 2k + 1)) for k = 1:nv])
    end
    samples = [[ch[i][idx[i]] for i = 1:nv] for idx in idxs]
    sort!(samples)
end

# The stabiliser H of point 1 among the permutations of the k points, acting on the
# pair variables (12, 13, 14, 23, 24, 34 for k = 4; 12, 13, 23 for k = 3).  With
# SYM = 1 each sum-of-squares term is summed over H, so the identity lies in the
# H-invariant polynomials and as many generic samples as their dimension suffice.
function stabiliser(k)
    if k == 4
        pm, _ = L.embed_s4_action()
        return [pm[p] for p in L.SymmetricGroup(4) if p.d[1] == 1]
    elseif k == 3
        return [L.Perm([1, 2, 3]), L.Perm([2, 1, 3])]
    end
end

loc(u, a, b) = (u - a) * (b - u)

function star_sos(k, d, B, BL, FF)
    nv = binomial(k, 2)
    R, x = polynomial_ring(FF, nv)
    sym = get(ENV, "SYM", "1") == "1"
    H = sym ? stabiliser(k) : [L.Perm(collect(1:nv))]
    nsamp = sym ? length(L.invariant_basis(nv, d, H)) : binomial(nv + d, nv)
    samples = grid_samples(nv, d, FF, nsamp)
    Rs = SampledMPolyRing(FF, samples)
    act(q, h) = evaluate(q, [x[h.d[i]] for i = 1:nv])
    idx = 1
    G = Matrix{typeof(R(1))}(undef, k, k)
    for i = 1:k
        G[i, i] = R(1)
        for j = i+1:k
            G[i, j] = G[j, i] = x[idx]; idx += 1
        end
    end
    long = [i == 1 for i = 1:k for j = i+1:k]  # pairs at point 1
    dom = [R(1)]
    for e = 1:nv
        push!(dom, loc(x[e], -1, long[e] ? BL : B))
    end
    if k >= 3
        push!(dom, L.simple_det(G))
    end
    if k == 4
        for S in IterTools.subsets(1:4, 3)
            push!(dom, L.simple_det(G[S, S]))
        end
    end
    sos = Dict()
    for (gi, g) in enumerate(dom)
        m = div(d - total_degree(g), 2)
        m < 0 && continue
        mons = monomials_upto(x, m)
        sos[(Symbol("star$k"), gi)] = LowRankMatPol([act(g, h) for h in H],
                                                    [[Rs(act(mon, h)) for mon in mons] for h in H])
    end
    sos, samples
end

function star_sos2(d, BL, FF)
    # as in the package: Chebyshev samples on [-1, BL]
    R, x = polynomial_ring(FF, 1)
    s1 = [[FF(floor(BigInt, y * 10^4) // 10^4)] for y in ClusteredLowRankSolver.sample_points_chebyshev(d, -1, BL)]
    sort!(s1)
    Rs = SampledMPolyRing(FF, s1)
    xs = Rs(x[1])
    ub = ClusteredLowRankSolver.basis_chebyshev(div(d, 2), xs)
    sos = Dict()
    sos[(:star2, 1)] = LowRankMatPol([1], [ub[1:div(d, 2)+1]])
    sos[(:star2, 2)] = LowRankMatPol([loc(x[1], -1, BL)], [ub[1:div(d, 2)]])
    sos, s1
end

function star_problem(n, d1, delta, B, BL; FF = QQ)
    t = 2
    d2 = delta
    representatives = [L.gram_matrix(m, FF) for m = 0:2t]
    orbits = representatives[1:t+1]
    irreps = [lambda for lambda in L.lambdas(2, d1) if lambda[1] != lambda[2] || iseven(lambda[1])]
    weven = [[k for k = 0:lambda[1]-lambda[2] if iseven(L.countels(lambda, 2, k))] for lambda in irreps]
    for li in eachindex(irreps)
        L.compute_PS(n, irreps[li], t; ws = weven[li])
    end
    all_matrices = []
    all_samples = []
    m1, s1 = L.make_sos(Val(1), 0, [-1, B], Dict(), FF)
    push!(all_matrices, m1); push!(all_samples, s1)
    m2, s2 = star_sos2(delta, BL, FF)
    push!(all_matrices, m2); push!(all_samples, s2)
    for k = 3:4
        mk, sk = star_sos(k, delta, B, BL, FF)
        push!(all_matrices, mk); push!(all_samples, sk)
        @info "k = $k: $(length(sk)) samples, $(length(mk)) sum-of-squares blocks"
    end
    d_tensor = [div(d2 - sum(lambda), 2) for lambda in irreps]
    atl = [[(size(orbits[i], 2), j, k) for i in eachindex(orbits) for (j, k) in L.admissible_tuples(irreps[li], size(orbits[i], 2), weven[li], d_tensor[li])] for li in eachindex(irreps)]
    t2i = [Dict(atl[li][k] => k for k in eachindex(atl[li])) for li in eachindex(irreps)]
    blocksizes = [length(t2i[li]) for li in eachindex(irreps)]
    constraints = [L.las2_constraint(n, t, k, d_tensor, representatives, FF, all_samples,
                       all_matrices, t2i, irreps, weven, blocksizes, true; sampled = true) for k = 1:2t]
    objective = Dict()
    mat = zeros(Int64, blocksizes[1], blocksizes[1]); mat[1, 1] = 1
    objective[[0, 0]] = mat
    Problem(false, Objective(0, objective, Dict()), constraints)
end

d1 = parse(Int, ARGS[1]); dl = parse(Int, ARGS[2]); prec = parse(Int, ARGS[3])
B = parse(Rational{BigInt}, ARGS[4]); BL = parse(Rational{BigInt}, ARGS[5])
out = length(ARGS) >= 6 ? ARGS[6] : "star_results.txt"
setprecision(prec)
t0 = time()
problem = star_problem(4, d1, dl, B, BL)
sdp = ClusteredLowRankSDP(problem)
@info "built in $(round(time() - t0, digits = 1)) s"
status, primal, dual, _, _ = solvesdp(convert_to_prec(sdp, prec); prec = prec,
    duality_gap_threshold = 1e-8, primal_error_threshold = 1e-8, dual_error_threshold = 1e-8,
    omega_p = big(10)^3, omega_d = big(10)^3)
val = objvalue(problem, dual)
open(out, "a") do f
    println(f, "d1=$d1 delta=$dl prec=$prec B=$B BL=$BL value=$(Float64(val)) status=$status seconds=$(round(time() - t0, digits = 1))")
end
println("value ", Float64(val))

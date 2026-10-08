# One part of the chunked construction of the star programme (star/star_las2.jl) for the
# fast solver of fast/: the second level of the hierarchy on the admissible sets whose pairs
# above B form a star, with pairs up to BL at the centre of the star.  Same chunk format as
# fast/build_part.jl, so fast/assemble.jl and fast/solve_fast.jl apply unchanged.
#   julia --project=$LSC -t 1 build_part_star.jl D1 DELTA PREC CHUNK B BL PART NP OUTDIR
# Part 0 builds constraints 1-3; part p >= 1 builds the chunks c of constraint 4 with
# c mod NP == p - 1.  Floating point; exploration only.
using LasserreSphericalCodes, ClusteredLowRankSolver, Nemo, Serialization, IterTools, Random
Random.seed!(1)
const L = LasserreSphericalCodes
const CL = ClusteredLowRankSolver
const Arblib = CL.Arblib
rss() = begin
    for l in eachline("/proc/self/status"); startswith(l, "VmRSS") && return parse(Int, split(l)[2]) / 1e6; end
end
report(tag) = (GC.gc(); ccall(:malloc_trim, Cvoid, (Cint,), 0);
    println("MEM $tag rss=$(round(rss(),digits=3)) GB maxrss=$(round(Sys.maxrss()/1e9,digits=3)) GB t=$(round(time()-T0,digits=1)) s"); flush(stdout))

sameval(a, b) = size(a) == size(b) && all(Arblib.equal(a[i, j], b[i, j]) for i = 1:size(a, 1), j = 1:size(a, 2))

restrict(p::CL.SampledMPolyRingElem, R, idx) = CL.SampledMPolyRingElem(R, p.evaluations[idx])
restrict(p, R, idx) = p
function restrict(M::CL.LowRankMatPol, R, idx)
    CL.LowRankMatPol([restrict(x, R, idx) for x in M.lambda], [[restrict(x, R, idx) for x in v] for v in M.vs],
                     [[restrict(x, R, idx) for x in v] for v in M.ws])
end

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


function build_part(n, B, BL, d1, delta, prec, chunk, part, np, outdir)
    t = 2; FF = QQ; d2 = delta
    Ns = [0, delta, delta, delta]
    representatives = [L.gram_matrix(m, FF) for m = 0:2t]
    orbits = representatives[1:t+1]
    irreps = [lambda for lambda in L.lambdas(2, d1) if lambda[1] != lambda[2] || iseven(lambda[1])]
    weven = [[k for k = 0:lambda[1]-lambda[2] if iseven(L.countels(lambda, 2, k))] for lambda in irreps]
    for li in eachindex(irreps)
        L.compute_PS(n, irreps[li], t; ws=weven[li])
    end
    empty!(L.memoize_cache(L.integrateorthogonal)); empty!(L.memoize_cache(L.integrateorthogonalmemoize))
    all_matrices = []; all_samples = []
    m1, s1 = L.make_sos(Val(1), 0, [-1, B], Dict(), FF, sampled=true)
    push!(all_matrices, m1); push!(all_samples, s1)
    m2, s2 = star_sos2(delta, BL, FF)
    push!(all_matrices, m2); push!(all_samples, s2)
    for k = 3:4
        mk, sk = star_sos(k, delta, B, BL, FF)
        push!(all_matrices, mk); push!(all_samples, sk)
    end
    @assert length(all_samples[1]) == 1
    report("sos parts ready (m2=$(length(s2)) m3=$(length(all_samples[3])) m4=$(length(all_samples[4])))")
    d_tensor = [div(d2 - sum(lambda), 2) for lambda in irreps]
    atl = [[(size(orbits[i], 2), j, k) for i in eachindex(orbits) for (j, k) in L.admissible_tuples(irreps[li], size(orbits[i], 2), weven[li], d_tensor[li])] for li in eachindex(irreps)]
    tuple_to_index = [Dict(atl[li][k] => k for k in eachindex(atl[li])) for li in eachindex(irreps)]
    blocksizes = [length(tuple_to_index[li]) for li in eachindex(irreps)]
    base = [0, 1, 1 + length(all_samples[2]), 1 + length(all_samples[2]) + length(all_samples[3])]
    for k = (part == 0 ? (1:3) : (4:4))
        m_k = length(all_samples[k])
        ranges = k == 4 ? [rg for (ci, rg) in enumerate(Iterators.partition(1:m_k, chunk)) if (ci - 1) % np == part - 1] :
                 k == 3 ? collect(Iterators.partition(1:m_k, chunk)) : [1:m_k]
        for rg in ranges
            fout = joinpath(outdir, "k$(k)_$(lpad(first(rg), 5, '0')).jls")
            isfile(fout) && continue
            Ablocks = Dict{Any, Dict{Int, Union{CL.LowRankMat, CL.ArbRefMatrix}}}()
            sizes = Dict{Any, Int}()
            cvals = Dict{Int, Arblib.Arb}()
            if k >= 3
                smp = all_samples[k][rg]
                Rs = CL.SampledMPolyRing(FF, smp)
                mats = Dict{Any, Any}(key => restrict(v, Rs, rg) for (key, v) in all_matrices[k])
            else
                smp = all_samples[k]
                mats = copy(all_matrices[k])
            end
            as = copy(all_samples); as[k] = smp
            am = copy(all_matrices); am[k] = mats
            con = L.las2_constraint(n, t, k, d_tensor, representatives, FF, as, am, tuple_to_index,
                                    irreps, weven, blocksizes, false; sampled=true)
            sdpk = ClusteredLowRankSDP(CL.Problem(false, CL.Objective(0, Dict(), Dict()), [con]); prec=prec)
            @assert length(sdpk.A) == 1
            offset = base[k] + first(rg) - 1
            for (l, nm) in enumerate(sdpk.matrix_coeff_names[1])
                d = get!(Ablocks, nm) do; Dict{Int, Union{CL.LowRankMat, CL.ArbRefMatrix}}(); end
                for (i, M) in sdpk.A[1][l][1, 1]
                    if M isa CL.LowRankMat && all(M.vs[r] === M.ws[r] || sameval(M.vs[r], M.ws[r]) for r in eachindex(M.vs))
                        M = CL.LowRankMat(M.lambda, M.vs, M.vs)      # one copy of the vectors
                    end
                    d[offset + i] = M
                end
                sizes[nm] = size(sdpk.C.blocks[1].blocks[l], 1)
            end
            for i = 1:size(sdpk.c[1], 1)
                cvals[offset + i] = Arblib.Arb(sdpk.c[1][i], prec=prec)
            end
            con = nothing; sdpk = nothing; mats = nothing
            serialize(fout * ".tmp", (Ablocks=Ablocks, sizes=sizes, cvals=cvals, m=sum(length, all_samples)))
            mv(fout * ".tmp", fout, force=true)
            Ablocks = nothing; cvals = nothing
            report("constraint $k samples $(first(rg))-$(last(rg))")
        end
    end
end
d1 = parse(Int, ARGS[1]); dl = parse(Int, ARGS[2]); prec = parse(Int, ARGS[3]); chunk = parse(Int, ARGS[4])
B = parse(Rational{BigInt}, ARGS[5]); BL = parse(Rational{BigInt}, ARGS[6])
part = parse(Int, ARGS[7]); np = parse(Int, ARGS[8]); out = ARGS[9]
setprecision(prec)
const T0 = time()
report("start part $part of $np, B=$B BL=$BL")
mkpath(out)
build_part(4, B, BL, d1, dl, prec, chunk, part, np, out)
report("part built")

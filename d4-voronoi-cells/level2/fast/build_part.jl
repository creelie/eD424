# One part of the chunked construction of the LLM24 second-level SDP on D = [-1, 1/2 + s]
# (see level2-fit/chunked_build.jl, whose build_chunked this follows line for line).
# Part 0 builds constraints 1-3; part p >= 1 builds the chunks c of constraint 4 with
# c mod NP == p - 1.  Every sample keeps the global index it has in the unchunked build,
# so merge_parts.jl can assemble the parts in any order.
#   julia --project=. -t 1 build_part.jl D1 DELTA PREC CHUNK s PART NP OUTDIR
# Each chunk is written to OUTDIR/k<k>_<first sample>.jls as soon as it is built (a chunk
# already on disk is skipped), which keeps the build processes small and lets a killed
# build resume.
using LasserreSphericalCodes, ClusteredLowRankSolver, Nemo, Serialization, IterTools
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

function build_part(n, D, d1, delta, prec, chunk, part, np, outdir)
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
    for k = 1:3
        m, s = L.make_sos(Val(k), Ns[k], D, Dict(), FF, sampled=true)
        push!(all_matrices, m); push!(all_samples, s)
    end
    sos4_poly, samples4 = L.make_sos(Val(4), Ns[4], D, Dict(), FF, sampled=false)
    push!(all_matrices, sos4_poly); push!(all_samples, samples4)
    report("sos parts ready (m4=$(length(samples4)))")
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
            if k == 4
                smp = all_samples[4][rg]
                mats = Dict{Any, Any}(key => L.sample_sos(v, smp, FF) for (key, v) in sos4_poly)
            elseif k == 3
                smp = all_samples[3][rg]
                Rs = CL.SampledMPolyRing(FF, smp)
                mats = Dict{Any, Any}(key => restrict(v, Rs, rg) for (key, v) in all_matrices[3])
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
s = parse(Rational{BigInt}, ARGS[5]); part = parse(Int, ARGS[6]); np = parse(Int, ARGS[7]); out = ARGS[8]
setprecision(prec)
const T0 = time()
report("start part $part of $np, s=$s")
mkpath(out)
build_part(4, [-1, 1//2 + s], d1, dl, prec, chunk, part, np, out)
report("part built")

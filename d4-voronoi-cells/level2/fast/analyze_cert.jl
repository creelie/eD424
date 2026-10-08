# Read the pair penalty P2 off a level-2 solution (checkpoint of patched_solver.jl).
#   julia --project=. analyze_cert.jl SDP.jls|CHUNKDIR CKPT.jls d s
# For every code C with inner products in [-1, 1/2+s]:  sum_{pairs of C} P2(u_ij) <= B - |C|,
# where B = K(0,0) and P2 is the degree-d sos multiplier of the pair constraint:
#   P2(u) = v(u)' Y[(:sos2,1)] v(u) + (u+1)(b-u) w(u)' Y[(:sos2,2)] w(u),  v, w Chebyshev T_0.. (b = 1/2+s).
using ClusteredLowRankSolver, Serialization, Printf
const CL = ClusteredLowRankSolver
const Arblib = CL.Arblib
setprecision(256)
src, ckf, d, s = ARGS[1], ARGS[2], parse(Int, ARGS[3]), parse(Rational{BigInt}, ARGS[4])
b = big(1)//2 + s
# block names in the order assemble.jl gives them (same files, same order, same Dict type)
function names_from_dir(dir)
    files = sort(filter(f -> endswith(f, ".jls"), readdir(dir, join=true)))
    Ab = Dict{Any, Dict{Int, Union{CL.LowRankMat, CL.ArbRefMatrix}}}(); cnt = Dict{Any, Int}()
    k2 = nothing
    for f in files
        r = deserialize(f)
        for (nm, dd) in r.Ablocks
            get!(Ab, nm) do; Dict{Int, Union{CL.LowRankMat, CL.ArbRefMatrix}}(); end
            cnt[nm] = get(cnt, nm, 0) + length(dd)
        end
        startswith(basename(f), "k2_") && (k2 = r)
        r = nothing; GC.gc()
    end
    for nm in collect(keys(Ab)); cnt[nm] == 0 && delete!(Ab, nm); end
    return collect(keys(Ab)), k2
end
if isdir(src)
    cache = rstrip(src, '/') * ".names.jls"   # the chunk files never change: read them once
    if isfile(cache)
        names, k2 = deserialize(cache)
    else
        names, k2 = names_from_dir(src); serialize(cache, (names, k2))
    end
else
    sdp = deserialize(src); names = sdp.matrix_coeff_names[1]; k2 = nothing
end
ck = deserialize(ckf)
@assert length(names) == length(ck.Y[1]) "names $(length(names)) vs blocks $(length(ck.Y[1]))"
blk(nm) = findfirst(==(nm), names)
tobig(M) = [BigFloat(M[i, j]) for i = 1:size(M, 1), j = 1:size(M, 2)]
Y1 = tobig(ck.Y[1][blk((:sos2, 1))]); Y2 = tobig(ck.Y[1][blk((:sos2, 2))]); Y0 = tobig(ck.Y[1][blk([0, 0])])
@assert size(Y1, 1) == d ÷ 2 + 1 && size(Y2, 1) == d ÷ 2
B = Y0[1, 1]
cheb(n, x) = CL.basis_chebyshev(n, x)
P2(u) = (v = cheb(d ÷ 2, u); w = cheb(d ÷ 2 - 1, u); v' * Y1 * v + (u + 1) * (b - u) * (w' * Y2 * w))
@printf("iteration %d   B = K(0,0) = %.10f   (B - 24 = %.6f)\n", ck.iter, B, B - 24)
# check against the pair-constraint matrices at the samples, when available
if k2 !== nothing
    mx = 0.0
    for (i, M) in k2.Ablocks[(:sos2, 1)]
        u = BigFloat(M.vs[1][2]) / BigFloat(M.vs[1][1])
        val = sum(BigFloat(M.lambda[r]) * (vec(tobig(M.vs[r]))' * Y1 * vec(tobig(M.vs[r]))) for r in eachindex(M.vs))
        M2 = get(k2.Ablocks[(:sos2, 2)], i, nothing)
        M2 !== nothing && (val += sum(BigFloat(M2.lambda[r]) * (vec(tobig(M2.vs[r]))' * Y2 * vec(tobig(M2.vs[r]))) for r in eachindex(M2.vs)))
        global mx = max(mx, Float64(abs(val - P2(u))))
    end
    @printf("P2 formula vs constraint matrices at the %d pair samples: max difference %.2e\n", length(k2.Ablocks[(:sos2, 1)]), mx)
end
roots = big.([-1, -1//2, 0, 1//2])
dist(u) = u >= big(1)//2 ? big(0) : minimum(abs.(u .- roots))   # [1/2, b] counts as the root 1/2
println("P2 at -1, -1/2, 0, 1/2, 1/2+s: ", join([@sprintf("%.4e", P2(u)) for u in [roots; b]], "  "))
D4sum = 12P2(big(-1)) + 96P2(big(-1)//2) + 72P2(big(0)) + 96P2(big(1)//2)
@printf("sum of P2 over the 276 pairs of D4 = %.6f  (must be <= B - 24 = %.6f)\n", D4sum, B - 24)
h = big(1)//10^6
@printf("slope at the end: P2'(b) = %.5f\n", (P2(b) - P2(b - h)) / h)
us = range(big(-1), b, length=30001)
vals = [P2(u) for u in us]
@printf("min of P2 on the grid: %.3e at u = %.4f\n", minimum(vals), us[argmin(vals)])
mins = [i for i = 2:length(us)-1 if vals[i] <= vals[i-1] && vals[i] <= vals[i+1]]
println("local minima of P2:"); for i in mins; @printf("   u = %+.5f   P2 = %.4e\n", us[i], vals[i]); end
rat = [(dist(u) > 1e-4 ? vals[i] / dist(u)^2 : big(Inf)) for (i, u) in enumerate(us)]
i = argmin(rat); @printf("c = min P2(u)/dist(u,roots)^2 = %.5f at u = %.4f\n", rat[i], us[i])
for (lo, hi) in [(-1, -0.75), (-0.75, -0.25), (-0.25, 0.25), (0.25, Float64(b))]
    sel = [j for (j, u) in enumerate(us) if lo <= u <= hi]
    j = sel[argmin(rat[sel])]; @printf("   near root in [%.2f, %.3f]: c = %.5f at u = %.4f\n", lo, hi, rat[j], us[j])
end
@printf("(B - 24)/c = %.5f   (the Gram route needs sum_{i<j} dist^2 < 0.2535 on 24-point codes)\n", (B - 24) / rat[i])
open(ckf * ".P2.txt", "w") do io
    for (u, v) in zip(us[1:10:end], vals[1:10:end]); @printf(io, "%.6f %.8e\n", u, v); end
end

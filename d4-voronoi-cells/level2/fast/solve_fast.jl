# julia --project=. -t 4 solve_fast.jl SDP.jls|CHUNKDIR PREC MAXIT MODE [LEVELS] [GAP]     MODE = tiled | fast
using ClusteredLowRankSolver, Serialization, LinearAlgebra
const CL = ClusteredLowRankSolver
rss() = begin
    for l in eachline("/proc/self/status"); startswith(l, "VmRSS") && return parse(Int, split(l)[2]) / 1e6; end
end
prec = parse(Int, ARGS[2]); maxit = parse(Int, ARGS[3]); mode = ARGS[4]
Base.eval(CL, :(using LinearAlgebra, Printf))
Base.eval(CL, Meta.parse("begin\n" * read(joinpath(@__DIR__, "patched_solver.jl"), String) * "\nend"))
Base.eval(CL, Meta.parse("begin\n" * read(joinpath(@__DIR__, "tiled_S.jl"), String) * "\nend"))
Base.eval(CL, Meta.parse("begin\n" * read(joinpath(@__DIR__, "fast_S.jl"), String) * "\nend"))
CL.FAST[] = mode == "fast"
length(ARGS) >= 5 && (CL.OZ_LEVELS[] = parse(Int, ARGS[5]))
gap = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 1e-8
haskey(ENV, "AUX_CHECK") && (CL.AUX_CHECK[] = parse(Int, ENV["AUX_CHECK"]))
haskey(ENV, "OZ_K") && (CL.OZ_K[] = parse(Int, ENV["OZ_K"]))
haskey(ENV, "CHOL_K") && (CL.CHOL_K[] = parse(Int, ENV["CHOL_K"]))
haskey(ENV, "CHOL_L") && (CL.CHOL_L[] = parse(Int, ENV["CHOL_L"]))
haskey(ENV, "OZ_TILE") && (CL.OZ_TILE[] = parse(Int, ENV["OZ_TILE"]))
haskey(ENV, "FAST_AUX") && (CL.FAST_AUX[] = ENV["FAST_AUX"] == "1")
CL.TILE[] = 1000; CL.GRAM_HIGHRANK[] = true
BLAS.set_num_threads(Threads.nthreads())
setprecision(prec)
include(joinpath(@__DIR__, "assemble.jl"))
sdp = isdir(ARGS[1]) ? assemble_sdp(sort(filter(f -> endswith(f, ".jls"), readdir(ARGS[1], join=true))), prec) : deserialize(ARGS[1])
GC.gc(); println("MEM loaded rss=$(rss()) GB threads=$(Threads.nthreads()) mode=$mode prec=$prec K=$(CL.OZ_K[]) levels=$(CL.OZ_LEVELS[]) cholK=$(CL.CHOL_K[]) cholL=$(CL.CHOL_L[])"); flush(stdout)
t0 = time()
status, psol, dsol, _t, _e = solvesdp(sdp; prec=prec, maxiterations=maxit, duality_gap_threshold=gap,
    primal_error_threshold=1e-8, dual_error_threshold=1e-8, omega_p=big(10)^parse(Int, get(ENV, "OMEGA_EXP", "3")), omega_d=big(10)^parse(Int, get(ENV, "OMEGA_EXP", "3")))
println("FPROF W=$(round(CL.FPROF[1],digits=1)) digits=$(round(CL.FPROF[2],digits=1)) gemm=$(round(CL.FPROF[3],digits=1)) comb=$(round(CL.FPROF[4],digits=1)) hrG=$(round(CL.FPROF[5],digits=1)) hrgram=$(round(CL.FPROF[6],digits=1)) final=$(round(CL.FPROF[7],digits=1)) aux=$(round(CL.FPROF[8],digits=1))")
println("MODE=$mode status=$status seconds=$(round(time()-t0,digits=1)) maxrss=$(round(Sys.maxrss()/1e9,digits=3)) GB")

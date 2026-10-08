# Entry point for build_part_star.jl that also runs on Windows: same arguments.  It replaces
# the two Linux-only calls (the resident set from /proc/self/status, and malloc_trim) by
# portable ones and runs the result.
#   julia --project=$LSC -t 1 pc_build_star.jl D1 DELTA PREC CHUNK B BL PART NP OUTDIR
const HERE = @__DIR__
src = read(joinpath(HERE, "build_part_star.jl"), String)
src = replace(src,
    "for l in eachline(\"/proc/self/status\"); startswith(l, \"VmRSS\") && return parse(Int, split(l)[2]) / 1e6; end" =>
        "return Sys.maxrss() / 1e9")
Sys.islinux() || (src = replace(src, "ccall(:malloc_trim, Cvoid, (Cint,), 0);" => ""))
@assert !occursin("/proc/self", src)
include_string(Main, src, joinpath(HERE, "build_part_star.jl"))

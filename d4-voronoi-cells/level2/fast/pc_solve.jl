# Entry point for solve_fast.jl that also runs on Windows: same arguments and environment
# variables.  It reads solve_fast.jl and the solver files it loads, replaces the two
# Linux-only calls (the resident set from /proc/self/status, and malloc_trim) by portable
# ones (Sys.maxrss, and nothing), and runs the result.
#   julia --project=$LSC -t 8 pc_solve.jl CHUNKDIR PREC MAXIT MODE [LEVELS] [GAP]
const HERE = @__DIR__

function portable(s)
    s = replace(s,
        "parse(Int, split(first(filter(l -> startswith(l, \"VmRSS\"), readlines(\"/proc/self/status\"))))[2]) / 1e6" =>
            "(Sys.maxrss() / 1e9)",
        "for l in eachline(\"/proc/self/status\"); startswith(l, \"VmRSS\") && return parse(Int, split(l)[2]) / 1e6; end" =>
            "return Sys.maxrss() / 1e9")
    Sys.islinux() ? s : replace(s, "malloc_trim(0)" => "nothing")
end

patched(name) = portable(read(joinpath(HERE, name), String))

src = portable(read(joinpath(HERE, "solve_fast.jl"), String))
for name in ("patched_solver.jl", "tiled_S.jl", "fast_S.jl")
    global src = replace(src, "read(joinpath(@__DIR__, \"$name\"), String)" => "patched(\"$name\")")
end
src = replace(src, "include(joinpath(@__DIR__, \"assemble.jl\"))" => "include(joinpath(HERE, \"assemble.jl\"))")
@assert !occursin("/proc/self", src) && !occursin("@__DIR__", src)
include_string(Main, src, joinpath(HERE, "solve_fast.jl"))

# Read the moment side x of a solved level-2 checkpoint and sum it by constraint type (subset size).
using ClusteredLowRankSolver, Serialization
const CL = ClusteredLowRankSolver
setprecision(128)
dir, ckf = ARGS[1], ARGS[2]
files = sort(filter(f -> endswith(f, ".jls"), readdir(dir, join=true)))
typ = Dict{Int,Int}()
for f in files
    r = deserialize(f); k = parse(Int, basename(f)[2:2])
    for i in keys(r.cvals); typ[i] = k; end
end
ck = deserialize(ckf)
println("checkpoint fields: ", fieldnames(typeof(ck)))
x = ck.x
xs = x isa Vector ? x[1] : x
m = length(typ)
println("m = $m, size x = ", size(xs))
for k = 1:4
    idx = [i for i = 1:m if typ[i] == k]
    s = sum(BigFloat(xs[i]) for i in idx)
    println("type $k: $(length(idx)) rows, sum x = $(Float64(s))")
end

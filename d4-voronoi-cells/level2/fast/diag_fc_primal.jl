# The pair side of the moment solution of the exactly-24 programme (checkpoint of patched_solver.jl):
#   julia --project=. diag_fc_primal.jl CHUNKDIR.names.jls CKPT.jls [N]
# In the sampled programme the moment vector x carries the pair measure at the samples u_i of the
# pair constraint (as x_i); S_k = N + 2 sum_i mu_i G_k(u_i) for k up to the pair degree.
using ClusteredLowRankSolver, Serialization, Printf
const CL = ClusteredLowRankSolver
setprecision(256)
cache, ckf = ARGS[1], ARGS[2]
N = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 24
names, k2 = deserialize(cache)
ck = deserialize(ckf)
println("checkpoint after iteration ", ck.iter)
d = k2.Ablocks[(:sos2, 1)]
rows = sort(collect(keys(d)))
us = [BigFloat(d[i].vs[1][2]) / BigFloat(d[i].vs[1][1]) for i in rows]
mu = [BigFloat(ck.x[i, 1]) for i in rows]
function gk(k, u)
    a, b = one(u), 2u
    k == 0 && return a
    for _ = 1:k-1; a, b = b, 2u * b - a; end
    return b / (k + 1)
end
@printf("pairs = %.6f  (binomial(%d,2) = %d)\n", sum(mu), N, binomial(N, 2))
println(" sample u      weight")
for (u, m) in zip(us, mu); @printf("%9.5f  %12.6f\n", u, m); end
w = [103, 191, 280, 230, 159] .// 963
S = [N + 2 * sum(mu .* gk.(k, us)) for k = 1:16]
for k = 1:16; @printf("S_%-2d = %10.5f\n", k, S[k]); end
@printf("sum_k c_k S_k (k = 1..5) = %.6f\n", sum(BigFloat(w[k]) * S[k] for k = 1:5))

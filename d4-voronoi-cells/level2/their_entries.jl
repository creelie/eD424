# Compute entries of P(S) with the compute_PS of de Laat, Leijenhorst and de
# Muinck Keizer, one signature at a time, in a clean copy of their package
# (no cache installed), for comparison with ours by compare_entries.py:
#   julia --project=. their_entries.jl "[4,0]" "[6,2]" ...
using LasserreSphericalCodes
const L = LasserreSphericalCodes
for lam in [eval(Meta.parse(a)) for a in ARGS]
    ws = [k for k=0:lam[1]-lam[2] if iseven(L.countels(lam, 2, k))]
    t = @elapsed L.compute_PS(4, lam, 2; ws=ws, verbose=false)
    println(lam, " ws=", ws, " ", round(t, digits=1), "s"); flush(stdout)
end

# The second-level bound of de Laat, Leijenhorst and de Muinck Keizer on the
# enlarged domain D = [-1, 1/2 + s], for several s, one line per s appended to
# OUT.  Run from the folder of LasserreSphericalCodes after setup_cache.sh:
#   julia --project=. -t 4 las2_slack.jl D1 DELTA PREC OUT s1 s2 ...
# with s given as rationals (0, 1//200, ...).  Floating point, exploration:
# a proof needs the rounding of save_solution and the check of verify.
using LasserreSphericalCodes, ClusteredLowRankSolver
d1 = parse(Int, ARGS[1]); dl = parse(Int, ARGS[2]); prec = parse(Int, ARGS[3])
out = ARGS[4]
for x in ARGS[5:end]
    s = parse(Rational{BigInt}, x)
    t0 = time()
    val = las2(4, [-1, 1//2 + s], d1; delta=dl, d2=dl, precision=prec, eps=1e-8,
               save_solution=false, omega_p=big(10)^3, omega_d=big(10)^3)[1]
    open(out, "a") do f
        println(f, "d1=$d1 delta=$dl prec=$prec s=$s B=$(Float64(val)) seconds=$(round(time()-t0, digits=1))")
    end
end

# The second level on D = [-1, 1/2 + s] with a margin in the two-point
# constraint.  With the bound K(empty, empty) fixed at N, maximise mu such that
#     A_2K({x,y}) + SOS_2(u) + mu * w(u) = 0,
#     w(u) = (u + 1) (u + 1/2)^2 u^2 (1/2 + s - u)   (>= 0 on D),
# the other three constraints as in LLM24.  For a 24-point code with inner
# products in D the chain of the certificate then gives
#     mu * sum over pairs of w(u_ij) <= N - 24,
# so every inner product has w(u) <= (N - 24)/mu: it lies near -1, -1/2, 0 or
# 1/2 + [0, s].  Floating point, exploration.
# NOTE: the normalised roots of D4 are such a code, with w = 3s/8 on each of
# their 96 pairs at 1/2, so (N - 24)/mu >= 36 s always.  Since max w < 0.019779
# at s = 0.008 (and max w < 36 s for every s > 5.2e-4), the bound excludes no
# inner product there at any degree.  See margin_floor_check.py.
#   julia --project=. -t 4 las2_margin.jl D1 DELTA PREC s N OUT
using LasserreSphericalCodes, ClusteredLowRankSolver, Nemo
d1 = parse(Int, ARGS[1]); dl = parse(Int, ARGS[2]); prec = parse(Int, ARGS[3])
s = parse(Rational{BigInt}, ARGS[4]); N = parse(Rational{BigInt}, ARGS[5]); out = ARGS[6]
b = 1//2 + s
setprecision(prec)
t0 = time()
prob = las2_problem(4, [-1, b], d1; delta=dl, d2=dl, obj=N)
R, (x,) = polynomial_ring(QQ, 1)
w = (x + 1) * (x + QQ(1,2))^2 * x^2 * (QQ(b) - x)
cons = copy(prob.constraints)
c2 = cons[2]
@assert length(c2.samples[1]) == 1   # the two-point constraint, one variable
cons[2] = Constraint(c2.constant, c2.matrixcoeff, Dict(:mu => w), c2.samples, c2.scalings)
prob2 = Problem(true, Objective(0, Dict(), Dict(:mu => 1)), cons)
sdp = ClusteredLowRankSDP(prob2)
sdp = convert_to_prec(sdp, prec)
status, psol, dsol, _, _ = solvesdp(sdp; prec=prec, duality_gap_threshold=1e-8,
    primal_error_threshold=1e-8, dual_error_threshold=1e-8,
    omega_p=big(10)^3, omega_d=big(10)^3)
mu = freevar(dsol, :mu)
open(out, "a") do f
    println(f, "d1=$d1 delta=$dl s=$s N=$N status=$status mu=$(Float64(mu)) (N-24)/mu=$(Float64((N-24)/mu)) seconds=$(round(time()-t0, digits=1))")
end
println("status ", status, " mu ", Float64(mu))

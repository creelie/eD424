# Installs and precompiles the dependencies of LasserreSphericalCodes (run with --project=$LSC).
using Pkg
Pkg.instantiate()
Pkg.precompile()
using LasserreSphericalCodes, ClusteredLowRankSolver
println("packages ready")

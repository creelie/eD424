#!/bin/sh
# Build and solve the star programme (build_part_star.jl) with the fast solver of ../fast.
#   LSC=/path/to/LasserreSphericalCodes sh run_star_fast.sh D1 DELTA B BL OUTDIR [THREADS]
# Writes OUTDIR (chunks), OUTDIR.part0.log, OUTDIR.part1.log, OUTDIR.solve.log.
HERE=$(cd "$(dirname "$0")" && pwd)
D1=$1; DL=$2; B=$3; BL=$4; OUT=$5; TH=${6:-2}
[ -n "$OUT" ] && [ -n "$LSC" ] || { echo "usage: LSC=... sh run_star_fast.sh D1 DELTA B BL OUTDIR [THREADS]"; exit 1; }
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
cd "$LSC" || exit 1
for p in 0 1; do
  julia --project=. -t 1 "$HERE/build_part_star.jl" "$D1" "$DL" 128 20 "$B" "$BL" $p 1 "$OUT" > "$OUT.part$p.log" 2>&1
done
CKPT=$OUT.ckpt.jls CKPT_EVERY=5 OZ_K=7 CHOL_K=8 CHOL_L=10 OMEGA_EXP=3 \
  julia --project=. -t "$TH" "$HERE/../fast/solve_fast.jl" "$OUT" 128 200 fast 7 1e-6 > "$OUT.solve.log" 2>&1

#!/bin/sh
# The same programme with exactly FC_N points (default 24) and at most binomial(FC_N, 2) + FC_EPS
# pairs, maximising the weighted design defect sum_k c_k S_k (weights FC_W, default those of the
# paper's proposition "No room from the design defects").
#   LSC=/path/to/LasserreSphericalCodes [FC_EPS=1//100] [THREADS=4] sh solve_fc.sh CHUNKDIR LOG CKPT
# Warm start from a checkpoint of solve.sh at a moderate mu (both sides feasible):
#   WARM=ckpt_copy.jls WARM_NAMES=CHUNKDIR.names.jls OMEGA_EXP=4 sh solve_fc.sh ...
# (the names cache is written by analyze_cert.jl; OMEGA_EXP sets the start of the one new
# block, so that its product is near the checkpoint's mu).
HERE=$(cd "$(dirname "$0")" && pwd)
[ $# -eq 3 ] && [ -n "$LSC" ] || { echo "usage: LSC=... sh solve_fc.sh CHUNKDIR LOG CKPT"; exit 1; }
DIR=$(cd "$1" && pwd); LOG=$(cd "$(dirname "$2")" && pwd)/$(basename "$2"); CK=$(cd "$(dirname "$3")" && pwd)/$(basename "$3")
cd "$LSC" || exit 1
CKPT=$CK CKPT_EVERY=1 OZ_K=7 CHOL_K=8 CHOL_L=10 OMEGA_EXP=${OMEGA_EXP:-10} FC_MODE=${FC_MODE:-eps} FC_EPS=${FC_EPS:-1//100} \
  julia --project=. -t "${THREADS:-4}" "$HERE/solve_fc.jl" "$DIR" 128 200 fast 7 1e-6 > "$LOG" 2>&1

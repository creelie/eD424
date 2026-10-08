#!/bin/sh
# Solve the programme of CHUNKDIR (128-bit working precision, Schur complement in triple-double
# arithmetic, starting scale 1e10, checkpoint after every iteration).
#   LSC=/path/to/LasserreSphericalCodes [THREADS=4] [WARM=ckpt.jls] sh solve.sh CHUNKDIR LOG CKPT
# To resume after an interruption, copy CKPT and pass the copy as WARM; rows of the new log
# restart at 1.
HERE=$(cd "$(dirname "$0")" && pwd)
[ $# -eq 3 ] && [ -n "$LSC" ] || { echo "usage: LSC=... sh solve.sh CHUNKDIR LOG CKPT"; exit 1; }
DIR=$(cd "$1" && pwd); LOG=$(cd "$(dirname "$2")" && pwd)/$(basename "$2"); CK=$(cd "$(dirname "$3")" && pwd)/$(basename "$3")
cd "$LSC" || exit 1
CKPT=$CK CKPT_EVERY=1 OZ_K=7 CHOL_K=8 CHOL_L=10 OMEGA_EXP=${OMEGA_EXP:-10} \
  julia --project=. -t "${THREADS:-4}" "$HERE/solve_fast.jl" "$DIR" 128 200 fast 7 1e-6 > "$LOG" 2>&1

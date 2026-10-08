#!/bin/sh
# Build the chunk files of the second-level programme on [-1, 1/2 + s] (five processes, one
# thread each; part 0 builds constraints 1-3, parts 1-4 share the chunks of constraint 4).
#   LSC=/path/to/LasserreSphericalCodes sh build.sh OUTDIR [D1 DELTA S]
# The package must hold the zonal cache of ../setup_cache.sh.  Defaults: (14, 16), s = 1/125.
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$1; D1=${2:-14}; DL=${3:-16}; S=${4:-1//125}
[ -n "$OUT" ] && [ -n "$LSC" ] || { echo "usage: LSC=... sh build.sh OUTDIR [D1 DELTA S]"; exit 1; }
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
cd "$LSC" || exit 1
for p in 0 1 2 3 4; do
  julia --project=. -t 1 "$HERE/build_part.jl" "$D1" "$DL" 128 20 "$S" $p 4 "$OUT" > "$OUT.part$p.log" 2>&1 &
done
wait

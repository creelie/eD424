#!/bin/sh
# Install the zonal matrices into a copy of LasserreSphericalCodes, so that its
# second-level programme runs without the three-day, 128 GB construction.
#   sh setup_cache.sh /path/to/LasserreSphericalCodes
# compute_PS skips an entry whose file cache/zonalstore/4/... exists, but it
# first asks for the integrand files cache/zonalstore/pol-2-...; the empty
# placeholders written here tell it that nothing is left to prepare.
set -e
dst="$1/cache/zonalstore"
mkdir -p "$dst"
tar xzf "$(dirname "$0")/zonalstore_4.tar.gz" -C "$dst"
for f in "$dst"/4/pol-4-2-*.txt; do
    b=$(basename "$f")
    touch "$dst/pol-2-${b#pol-4-2-}"
done
echo "$(ls "$dst"/4 | wc -l) zonal entries installed in $dst/4"

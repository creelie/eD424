#!/bin/sh
# Unpack the deposited certificate and run the verification described in the
# paper (Section "The certificate, checked").  Needs python3 with python-flint
# (FLINT and Arb), gcc with GMP for the zonal kernel, and Lean 4 for the last
# step.  Steps 3 and 5 take about five hours on two cores.
set -e
cd "$(dirname "$0")"
if [ ! -f LasserreSphericalCodes.zip ]; then python3 fetch_certificate.py --join; fi
python3 fetch_certificate.py --check
rm -rf proofs && mkdir -p proofs && unzip -q LasserreSphericalCodes.zip -d proofs
DATA=$(find proofs -type d -name 4_24 | head -1)
echo "certificate data: $DATA"
( cd ../../multi_cap && python3 llm24_certificate_check.py "$OLDPWD/$DATA" )
( cd ../../zonal && gcc -O2 -o psker psker.c -lgmp && python3 o4.py && python3 ps_build.py spec.bin && ./psker spec.bin ps.txt && python3 verify45.py "$OLDPWD/$DATA" ps.txt )
( cd ../../lean && lean D4InnerProducts.lean )
echo "all steps passed"

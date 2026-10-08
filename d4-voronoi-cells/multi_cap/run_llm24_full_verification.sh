#!/usr/bin/env bash
# run_llm24_full_verification.sh -- the one computation this package does not
# contain: the full verification of the certificate of de Laat, Leijenhorst
# and de Muinck Keizer (doi:10.4121/74ce1c25-6fca-4680-8a36-e9c18e7e9594),
# including the construction of the zonal matrices and the four polynomial
# identities (steps 3 and 5 of their procedure), which needs about three days
# and 128 GB of memory, followed by the independent partial check of this
# package (llm24_certificate_check.py) on the same data.
#
# Requirements: a Linux machine with >= 128 GB of memory and 8 cores, about
# 20 GB of free disk, network access, and LasserreSphericalCodes.zip in the
# current directory (md5 02acd5270f7b3fa799abdeb5291706fd).  Everything else
# is installed by the script.  Run it inside tmux or nohup; it writes
# llm24_full_verification.log and llm24_partial_check.log next to itself.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="$HERE/llm24_full_verification.log"

echo "== $(date) : checking the data set" | tee "$LOG"
echo "02acd5270f7b3fa799abdeb5291706fd  LasserreSphericalCodes.zip" | md5sum -c - | tee -a "$LOG"
unzip -q -o LasserreSphericalCodes.zip

echo "== $(date) : installing Julia 1.10" | tee -a "$LOG"
if ! command -v julia >/dev/null 2>&1; then
  curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel 1.10
  export PATH="$HOME/.juliaup/bin:$PATH"
fi
julia --version | tee -a "$LOG"

cd LasserreSphericalCodes
echo "== $(date) : instantiating the package" | tee -a "$LOG"
julia --project=. -e 'using Pkg; Pkg.instantiate()' 2>&1 | tee -a "$LOG"

echo "== $(date) : full verification (zonal matrices, identities, objective, inner products)" | tee -a "$LOG"
/usr/bin/time -v julia --project=. -t 8 -e 'using LasserreSphericalCodes; ok = verify("proofs/4_24"); println("verify returned: ", ok); exit(ok === false ? 1 : 0)' 2>&1 | tee -a "$LOG"
echo "== $(date) : full verification finished" | tee -a "$LOG"

cd "$HERE"
echo "== $(date) : independent partial check of this package on the same data" | tee -a "$LOG"
python3 -m pip install --quiet python-flint
python3 "$HERE/llm24_certificate_check.py" "$HERE/../LasserreSphericalCodes/proofs/4_24" 2>&1 | tee "$HERE/llm24_partial_check.log"
echo "== $(date) : done; logs: $LOG and $HERE/llm24_partial_check.log" | tee -a "$LOG"

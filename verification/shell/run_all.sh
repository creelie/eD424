#!/usr/bin/env bash
# run_all.sh -- run every independent check of the finite computation in
# Proposition 5.1 and Proposition 5.4 of the paper.
#
#   verification/shell/run_all.sh          C, Julia, Lean and Python checks of
#                                          the certificates in verification/data
#   verification/shell/run_all.sh --full   also recompute R(4) from scratch
#                                          (ryshkov2.py, cddlib) and the
#                                          certificates (certify.py), and
#                                          compare them with verification/data
#
# Tools: gcc, python3 (sympy; pycddlib for --full), julia >= 1.6, lean 4.
# Set JULIA or LEAN to point at a binary that is not on PATH.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$HERE/data"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

JULIA="${JULIA:-$(command -v julia || true)}"
LEAN="${LEAN:-$(command -v lean || echo "$HOME/.elan/bin/lean")}"
FULL=0
[ "${1:-}" = "--full" ] && FULL=1

pass() { printf '  [ok] %s\n' "$1"; }
fail() { printf '  [FAIL] %s\n' "$1"; exit 1; }

echo "== C: vertices, minimal vectors, extreme rays, edges, orbits"
gcc -O2 -Wall -o "$WORK/check_polyhedron" "$HERE/c/check_polyhedron.c" -lm
"$WORK/check_polyhedron" "$DATA" | tee "$WORK/c.log" | tail -3
grep -q "ALL CHECKS PASSED" "$WORK/c.log" && pass "C" || fail "C"

echo "== Julia: candidates and det Q at each (Sturm sequences)"
[ -x "$JULIA" ] || fail "julia not found (set JULIA)"
"$JULIA" "$HERE/julia/check_candidates.jl" "$DATA" | tee "$WORK/jl.log" | tail -3
grep -q "ALL CHECKS PASSED" "$WORK/jl.log" && pass "Julia" || fail "Julia"

echo "== Lean: certificates checked by evaluation, and the sign lemma"
cp "$HERE/lean/R4Check.lean" "$WORK/committed.lean"
python3 "$HERE/python/gen_lean.py" > /dev/null
cmp -s "$HERE/lean/R4Check.lean" "$WORK/committed.lean" \
  && pass "lean/R4Check.lean matches verification/data" \
  || { cp "$WORK/committed.lean" "$HERE/lean/R4Check.lean"; fail "lean/R4Check.lean is out of date (run python/gen_lean.py)"; }
[ -x "$LEAN" ] || fail "lean not found (set LEAN)"
"$LEAN" "$HERE/lean/R4Check.lean" && pass "Lean: 8 theorems" || fail "Lean"

echo "== Python: candidates and det Q at each (sympy, exact)"
python3 "$HERE/python/edges.py" "$DATA/r4_vertices_edges.json" | tee "$WORK/py.log" | tail -3
grep -q "ALL CHECKS PASSED" "$WORK/py.log" && pass "Python" || fail "Python"

if [ "$FULL" = 1 ]; then
  echo "== Python: recompute R(4) and the certificates from scratch"
  python3 "$HERE/python/ryshkov2.py" "$WORK/r4.json" > "$WORK/r4.log"
  python3 - "$WORK/r4.json" "$DATA/r4_vertices_edges.json" <<'PY'
import json, sys
a, b = (json.load(open(f)) for f in sys.argv[1:3])
assert a == b, "recomputed R(4) differs from verification/data"
PY
  pass "ryshkov2.py reproduces verification/data/r4_vertices_edges.json"
  mkdir "$WORK/cert"
  python3 "$HERE/python/certify.py" "$WORK/r4.json" "$WORK/cert" > "$WORK/cert.log"
  for f in vertices.txt edges.txt unbounded_orbits.txt; do
    cmp -s "$WORK/cert/$f" "$DATA/$f" && pass "certify.py reproduces data/$f" || fail "data/$f differs"
  done
fi

echo "ALL CHECKS PASSED"

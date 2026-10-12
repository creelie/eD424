#!/usr/bin/env bash
# check.sh -- rebuild the Lean certificate of Proposition 5.1 and check it in the Lean kernel.
#
#   verification/lean_kernel/check.sh          regenerate cert/ and lean/D4Lean/{Cert,Data} from
#                                              data/, compare them with the committed files, build
#                                              the Lean project and print the axioms of the main
#                                              theorem
#   verification/lean_kernel/check.sh --full   first recompute data/ from scratch (enumerate.py,
#                                              equality.py) and compare it with the committed files
#
# Tools: python3 (sympy and numpy; pycddlib for cert/ and --full), elan (Lean 4 and Lake). The
# first build downloads Mathlib and its compiled files (lake exe cache get).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export PATH="$HOME/.elan/bin:$PATH"
FULL=0
[ "${1:-}" = "--full" ] && FULL=1

pass() { printf '  [ok] %s\n' "$1"; }
fail() { printf '  [FAIL] %s\n' "$1"; exit 1; }

if [ "$FULL" = 1 ]; then
  echo "== Python: recompute the vertices of R and the equality points"
  python3 "$HERE/compute/enumerate.py" "$WORK/vertices.json" | tail -1
  cmp -s "$WORK/vertices.json" "$HERE/data/vertices.json" \
    && pass "enumerate.py reproduces data/vertices.json" || fail "data/vertices.json differs"
  python3 "$HERE/compute/equality.py" "$HERE/data/vertices.json" "$WORK/eq.json" > /dev/null
  cmp -s "$WORK/eq.json" "$HERE/data/equality_points.json" \
    && pass "equality.py reproduces data/equality_points.json" || fail "data/equality_points.json differs"
fi

echo "== Python: the vertices are the classes of Table 5.1"
python3 "$HERE/compute/table.py" "$HERE/data/vertices.json" "$HERE/../data/vertices.txt" | tail -1

echo "== Python: the certificate and the Lean data"
python3 "$HERE/cert/make_cert.py" "$HERE/data/vertices.json" "$HERE/data/equality_points.json" \
  "$WORK/certificate.txt"
cmp -s "$WORK/certificate.txt" "$HERE/cert/certificate.txt" \
  && pass "make_cert.py reproduces cert/certificate.txt" || fail "cert/certificate.txt differs"
python3 "$HERE/lean/generate/make_lean.py" "$HERE/cert/certificate.txt" \
  "$HERE/data/equality_points.json" "$WORK/lean" 2> "$WORK/gen.log"
for d in Cert Data; do
  diff -r -q "$WORK/lean/D4Lean/$d" "$HERE/lean/D4Lean/$d" > /dev/null \
    && pass "make_lean.py reproduces lean/D4Lean/$d" || fail "lean/D4Lean/$d differs"
done

echo "== Lean: no sorry, no native_decide, no new axioms"
if grep -rn -E '\b(sorry|admit|native_decide)\b|^axiom |implemented_by|@\[extern' \
    "$HERE/lean/D4Lean" "$HERE/lean/D4Lean.lean"; then
  fail "forbidden construct in lean/"
fi
pass "lean/ uses none of sorry, admit, native_decide, axiom, implemented_by, extern"

echo "== Lean: build (every theorem checked by the kernel)"
cd "$HERE/lean"
lake exe cache get > /dev/null
lake build 2>&1 | tail -1
lake env lean D4Lean/Axioms.lean | tee "$WORK/axioms.txt"
grep -qx "'D4.certificate' depends on axioms: \[propext, Classical.choice, Quot.sound\]" \
  "$WORK/axioms.txt" && pass "D4.certificate uses only propext, Classical.choice, Quot.sound" \
  || fail "unexpected axioms"

echo "ALL CHECKS PASSED"

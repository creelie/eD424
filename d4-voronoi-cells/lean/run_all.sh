#!/bin/sh
# Check the fifteen standalone Lean files with the pinned toolchain and print
# the axiom report of each.  Needs `lean` on the PATH (elan installs the
# version named in lean-toolchain).  The four Lake projects are built
# separately:  (cd cell600 && lake build),  (cd certificate && lake build),
# (cd cardinality && lake build)  and  (cd count && lake build).
set -e
cd "$(dirname "$0")"
for f in D4Stress D4Meet D4Certificate D4InnerProducts D4RootLattices D4Closure D4SecondOrder D4HexagonLoop D4SecondCode D4NearContact D4Rigidity D4Cap D4HoleBudget D4DesignBudget D4A18; do
  echo "== $f.lean"
  lean "$f.lean"
done
echo "all fifteen files check"

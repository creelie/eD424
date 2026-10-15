#!/usr/bin/env bash
# build_submission.sh -- the source files for a journal submission, in
# build/submission/ (not committed):
#
#   Musin24CellConjectureTwoTranslates_final2.pdf         the compiled manuscript
#   Musin24CellConjectureTwoTranslates_final2-source.zip  main.tex with the figures
#                                                         named Fig1, Fig2, Fig3 as
#                                                         PDF (used by pdflatex) and
#                                                         EPS (vector artwork)
#
# Runs scripts/build_paper.sh first. Needs pdftops (poppler) for the EPS files.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME=Musin24CellConjectureTwoTranslates
OUT="$ROOT/build/submission"
"$ROOT/scripts/build_paper.sh" > /dev/null

rm -rf "$OUT"
mkdir -p "$OUT/src"
cd "$ROOT/paper"
sed -e 's#{figures/translates}#{Fig1}#' -e 's#{figures/descent}#{Fig2}#' \
    -e 's#{figures/graph}#{Fig3}#' main.tex > "$OUT/src/main.tex"
for i in 1 2 3; do grep -q "{Fig$i}" "$OUT/src/main.tex"; done
i=1
for f in translates descent graph; do
  cp "figures/$f.pdf" "$OUT/src/Fig$i.pdf"
  pdftops -eps "figures/$f.pdf" "$OUT/src/Fig$i.eps"
  i=$((i + 1))
done

# the zipped source must compile on its own
( cd "$OUT/src" \
  && pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null \
  && pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null \
  && rm -f main.aux main.log main.out main.pdf )
( cd "$OUT/src" && zip -q "$OUT/${NAME}_final2-source.zip" main.tex Fig1.pdf Fig2.pdf Fig3.pdf Fig1.eps Fig2.eps Fig3.eps )
rm -rf "$OUT/src"
cp "$ROOT/dist/${NAME}_final2.pdf" "$OUT/"
ls -l "$OUT"

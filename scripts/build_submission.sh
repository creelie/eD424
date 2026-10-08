#!/usr/bin/env bash
# build_submission.sh -- the source files for a journal submission, in
# build/submission/ (not committed):
#
#   densest-two-translates-d4.pdf          the compiled manuscript
#   densest-two-translates-d4-source.zip   main.tex with the figures named
#                                          Fig1, Fig2 as PDF (used by
#                                          pdflatex) and EPS (vector artwork)
#
# Runs scripts/build_paper.sh first. Needs pdftops (poppler) for the EPS files.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME=densest-two-translates-d4
OUT="$ROOT/build/submission"
"$ROOT/scripts/build_paper.sh" > /dev/null

rm -rf "$OUT"
mkdir -p "$OUT/src"
cd "$ROOT/paper"
sed -e 's#{figures/descent}#{Fig1}#' -e 's#{figures/graph}#{Fig2}#' main.tex > "$OUT/src/main.tex"
grep -q '{Fig1}' "$OUT/src/main.tex" && grep -q '{Fig2}' "$OUT/src/main.tex"
cp figures/descent.pdf "$OUT/src/Fig1.pdf"
cp figures/graph.pdf "$OUT/src/Fig2.pdf"
pdftops -eps figures/descent.pdf "$OUT/src/Fig1.eps"
pdftops -eps figures/graph.pdf "$OUT/src/Fig2.eps"

# the zipped source must compile on its own
( cd "$OUT/src" \
  && pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null \
  && pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null \
  && rm -f main.aux main.log main.out main.pdf )
( cd "$OUT/src" && zip -q "$OUT/$NAME-source.zip" main.tex Fig1.pdf Fig2.pdf Fig1.eps Fig2.eps )
rm -rf "$OUT/src"
cp "$ROOT/dist/$NAME.pdf" "$OUT/"
ls -l "$OUT"

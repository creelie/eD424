#!/usr/bin/env bash
# build_paper.sh -- build the paper and its submission files into dist/.
#
#   dist/densest-two-translates-d4.pdf          the compiled paper
#   dist/densest-two-translates-d4-tex.zip      main.tex with the figures as PNG
#                                               (and their TikZ sources)
#   dist/densest-two-translates-d4-arxiv.tar.gz main.tex and the figures as PDF,
#                                               ready for arXiv
#
# Needs pdflatex with amsart, tikz, tikz-3dplot and hyperref; pdftoppm
# (poppler) for the PNG figures; zip and tar.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER="$ROOT/paper"
DIST="$ROOT/dist"
NAME=densest-two-translates-d4
mkdir -p "$DIST"

latex() { pdflatex -interaction=nonstopmode -halt-on-error "$@" > /dev/null; }

echo "== figures"
cd "$PAPER/figures"
for f in descent graph; do
  latex "$f.tex"
  pdftoppm -png -r 200 -singlefile "$f.pdf" "$f"
  rm -f "$f.aux" "$f.log"
done

echo "== paper"
cd "$PAPER"
latex main.tex
latex main.tex
if grep -q "Overfull\|undefined" main.log; then
  grep "Overfull\|undefined" main.log
  echo "build_paper.sh: fix the warnings above" >&2
  exit 1
fi
cp main.pdf "$DIST/$NAME.pdf"

echo "== tex.zip (figures as PNG)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/$NAME/figures"
sed 's#\\includegraphics\(\[[^]]*\]\)\?{figures/\([a-z]*\)}#\\includegraphics\1{figures/\2.png}#' \
  main.tex > "$STAGE/$NAME/main.tex"
cp figures/descent.png figures/graph.png figures/descent.tex figures/graph.tex "$STAGE/$NAME/figures/"
( cd "$STAGE/$NAME" && latex main.tex && latex main.tex && rm -f main.aux main.log main.out main.pdf )
rm -f "$DIST/$NAME-tex.zip"
( cd "$STAGE" && zip -qr "$DIST/$NAME-tex.zip" "$NAME" )

echo "== arXiv tarball (figures as PDF)"
mkdir -p "$STAGE/arxiv/figures"
cp main.tex "$STAGE/arxiv/"
cp figures/descent.pdf figures/graph.pdf "$STAGE/arxiv/figures/"
( cd "$STAGE/arxiv" && latex main.tex && latex main.tex && rm -f main.aux main.log main.out main.pdf )
tar -czf "$DIST/$NAME-arxiv.tar.gz" -C "$STAGE/arxiv" .

rm -f main.aux main.log main.out
ls -l "$DIST"

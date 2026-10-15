#!/usr/bin/env bash
# build_paper.sh -- build the paper and the release files.
#
#   dist/Musin24CellConjectureTwoTranslates_final2.pdf          the paper
#   dist/Musin24CellConjectureTwoTranslates_final2-tex.zip      main.tex with the figures as
#                                                               PNG (and their TikZ sources)
#   dist/Musin24CellConjectureTwoTranslates_final2-arxiv.tar.gz main.tex and the figures as
#                                                               PDF, ready for arXiv
#   preprintMusin24CellConjectureTwoTranslates/Musin24CellConjectureTwoTranslates.pdf
#                                                               the paper (the only file there)
#   MasterMusin24CellConjectureTwoTranslates_masterzipfinal2.zip
#                                                               every tracked file with dist/
#                                                               and the preprint folder
#
# Needs pdflatex with amsart, tikz, tikz-3dplot and hyperref; pdftoppm
# (poppler) for the PNG figures; git, zip and tar.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
for t in pdflatex pdftoppm git zip tar; do
  command -v "$t" >/dev/null || { echo "missing tool: $t" >&2; exit 2; }
done

NAME=Musin24CellConjectureTwoTranslates
PRE=preprint$NAME
MASTER=Master${NAME}_masterzipfinal2.zip
FIGS="translates descent graph"

latex() { pdflatex -interaction=nonstopmode -halt-on-error "$@" > /dev/null; }

echo "== figures"
(
  cd paper/figures
  for f in $FIGS; do
    latex "$f.tex"
    pdftoppm -png -r 200 -singlefile "$f.pdf" "$f"
    rm -f "$f.aux" "$f.log"
  done
)

echo "== paper"
(
  cd paper
  rm -f main.aux main.out main.log
  for run in 1 2 3 4; do
    latex main.tex
    grep -q "Rerun to get" main.log || break
  done
)
if grep -E "Overfull|Underfull|undefined|Rerun to get" paper/main.log; then
  echo "build_paper.sh: fix the warnings above" >&2
  exit 1
fi
rm -f paper/main.aux paper/main.out paper/main.log

rm -rf dist
mkdir -p dist "$PRE"
cp paper/main.pdf "dist/${NAME}_final2.pdf"
cp paper/main.pdf "$PRE/$NAME.pdf"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "== tex zip (figures as PNG)"
SRC="$STAGE/${NAME}_final2-tex"
mkdir -p "$SRC/figures"
sed 's#\\includegraphics\(\[[^]]*\]\)\?{figures/\([a-z]*\)}#\\includegraphics\1{figures/\2.png}#' \
  paper/main.tex > "$SRC/main.tex"
for f in $FIGS; do cp "paper/figures/$f.png" "paper/figures/$f.tex" "$SRC/figures/"; done
( cd "$SRC" && latex main.tex && latex main.tex && rm -f main.aux main.log main.out main.pdf )
( cd "$STAGE" && zip -qrX "${NAME}_final2-tex.zip" "${NAME}_final2-tex" )
mv "$STAGE/${NAME}_final2-tex.zip" dist/

echo "== arXiv tarball (figures as PDF)"
mkdir -p "$STAGE/arxiv/figures"
cp paper/main.tex "$STAGE/arxiv/"
for f in $FIGS; do cp "paper/figures/$f.pdf" "$STAGE/arxiv/figures/"; done
( cd "$STAGE/arxiv" && latex main.tex && latex main.tex && rm -f main.aux main.log main.out main.pdf )
tar -czf "dist/${NAME}_final2-arxiv.tar.gz" -C "$STAGE/arxiv" .

echo "== $MASTER"
mkdir -p "$STAGE/master/Master$NAME"
git ls-files -z | grep -zvE "^(dist/|$PRE/|$MASTER\$)" \
  | while IFS= read -r -d '' f; do
      [ -e "$f" ] || continue
      mkdir -p "$STAGE/master/Master$NAME/$(dirname "$f")"
      cp -p "$f" "$STAGE/master/Master$NAME/$f"
    done
mkdir -p "$STAGE/master/Master$NAME/dist" "$STAGE/master/Master$NAME/$PRE"
cp dist/* "$STAGE/master/Master$NAME/dist/"
cp "$PRE/$NAME.pdf" "$STAGE/master/Master$NAME/$PRE/"
rm -f "$MASTER"
( cd "$STAGE/master" && zip -qrX "$ROOT/$MASTER" "Master$NAME" )

echo "built: $(ls dist | tr '\n' ' ')$PRE/$NAME.pdf $MASTER"

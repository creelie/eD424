# Target journal: Discrete & Computational Geometry

| Item | Details |
|---|---|
| Journal | Discrete & Computational Geometry (DCG) |
| Publisher | Springer Nature (Springer US) |
| ISSN | 0179-5376 (print), 1432-0444 (electronic) |
| Homepage | https://link.springer.com/journal/454 |
| Submission | Editorial Manager, https://www.editorialmanager.com/dcge |
| Manuscript | `dist/densest-two-translates-d4.pdf` (13 pages, `amsart`) |
| Source files | `scripts/build_submission.sh` writes `build/submission/`: the PDF and a source zip with `main.tex` and the figures as Fig1, Fig2 (PDF and EPS) |

## Scope fit

DCG publishes research in which geometry plays a central role, and sphere
packing is a standing topic of the journal. The enumeration of two-periodic
packings that the paper makes exact appeared there: A. Andreanov and
Y. Kallus, *Locally optimal 2-periodic sphere packings*, Discrete Comput.
Geom. 63 (2020), 182-208. The paper proves that no packing of unit balls whose
centres form two translates of a lattice in R^4 is denser than D4, the case of
Musin's 24-cell conjecture for such packings, with the only computational
statement checked by four independent programs and proved in Lean 4 with every
step checked by the kernel.

MSC 2020: primary 52C17; secondary 11H31, 11H55.

## Length and format

- 13 pages; the journal sets no page limit.
- One source file `main.tex` with the references in `thebibliography`.
- Title page: title, author, affiliation, e-mail address and ORCID. Six
  keywords (the guidelines ask for 4 to 6) and the MSC codes are in the
  first-page footnote.
- The abstract has 180 words; Springer asks for 150 to 250.
- The section "Statements and Declarations" has funding, competing interests,
  and data and code availability, as the guidelines require.
- Supplementary material: the Zenodo record of the repository,
  doi:10.5281/zenodo.23239767 (all versions), cited in the paper.

## Policy on AI assistance

- Springer Nature: large language models do not satisfy its authorship
  criteria, and their use is documented in the Methods section or a suitable
  alternative part of the manuscript (DCG submission guidelines,
  https://link.springer.com/journal/454/submission-guidelines).
- Appendix A of the paper says that the author used Claude (Anthropic) for
  programming, computation and drafting, checked the mathematics and is
  responsible for the content.

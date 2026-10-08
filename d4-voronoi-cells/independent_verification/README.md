# independent_verification/

A re-verification of this package and of the manuscript, run on
24 September 2026 from a fresh clone of v1.3.0 to v1.6.0, and kept as a
record of that check: section and statement numbers here are those of the
version checked, not of the current paper.  Start with `REPORT.md`.

| path | what it is |
| --- | --- |
| `REPORT.md` | the verdict, every check and its result, the findings, and the changes made to the package |
| `proof_gaps/` | the two gaps found in the reduction of Section 2 (Lemmas 2.7 and 2.9), with scripts that exhibit them and their output |
| `logs/lean/` | `run_all.sh`, the two Lake builds, and the recheck of `D4Stress.lean` after its comment was corrected |
| `logs/scripts_named/` | every script the paper names, one log each, and `SUMMARY.txt` |
| `logs/scripts_other/` | every other script of the package, with a wall-clock limit per script |
| `logs/searches/` | the long exploratory searches, run with longer limits |
| `logs/zonal/` | steps 3 and 5 of the certificate for 24 contacts, and the checks of the zonal matrices |
| `logs/figures/` | the figure scripts under two matplotlib versions, and the pixel comparison |
| `logs/paper/` | the LaTeX build and its comparison with the shipped PDF |
| `compare_figures.py` | pixel comparison of `paper/figures/*.png` with the committed files |
| `gegenbauer_check.py` | the check of `zonal/README.md` item 3, which the package describes but did not contain |
| `numscan.py` | lists the numbers quoted in the paper that no log or script of the package contains |
| `check_bibliography.py` | checks that every entry is cited, every citation is defined and every DOI link equals the printed DOI; `--online` compares titles and years with Crossref and DataCite |
| `logs/doi_audit_2026-09-25.md` | every DOI of the bibliography compared with the publisher's or indexer's page (v1.6.0) |

## Reproducing

The checks were run in a scratch clone, so that no script could overwrite a
shipped file, and each script was run from its own directory.  In outline:

    cd lean && sh run_all.sh
    (cd lean/cell600 && lake build) ; (cd lean/certificate && lake build)
    (cd lean && python3 gen_certificate_lean.py && python3 gen_innerproducts_lean.py && python3 gen_cell600_lean.py)
    (cd lean/certificate && python3 gen_data.py) ; git status   # must be clean
    (cd third_party/llm24-certificate && python3 fetch_certificate.py --join)
    #   unzip into proofs/ and run multi_cap/llm24_certificate_check.py on proofs/.../4_24
    (cd zonal && gcc -O2 -o psker psker.c -lgmp && python3 o4.py && python3 ps_build.py spec.bin)
    #   python3 spec_split.py spec.bin - specA.bin specB.bin ; ./psker specA.bin psA.txt & ./psker specB.bin psB.txt
    #   cat psA.txt psB.txt > ps.txt
    #   python3 verify45.py DATA ps.txt 1,2 ; python3 verify45.py DATA ps.txt 3
    #   python3 verify45.py DATA ps.txt 4 zonal work4z/z4.pkl
    #   SOS_ONLY="s28 m22 b3 b4 s0 s5" bash run_sos4.sh DATA ps.txt work4a
    #   SOS_ONLY="b2 b19 b20 b21 s6 m13 m45 b1" bash run_sos4.sh DATA ps.txt work4b
    #   python3 combine4.py work4a/s4_total.pkl work4b/s4_total.pkl work4z/z4.pkl
    python3 independent_verification/gegenbauer_check.py zonal zonal/ps.txt
    (cd paper && latexmk -pdf D4.tex)

The figure comparison needs matplotlib 3.10.9, the version the shipped
figures were drawn with.  Run `python3 fig_*.py` in `paper/figures_new/`,
then `python3 independent_verification/compare_figures.py` from the
repository root.

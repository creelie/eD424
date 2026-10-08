# The certificate of de Laat, Leijenhorst and de Muinck Keizer

The proof of the twenty-four-contact case (Theorem "Twenty-four points" of
the paper, Section "The certificate kernel") consumes one finite object: the
exact rational feasible point of the second level of the Lasserre hierarchy
for spherical codes in S^3 with maximal inner product 1/2, found by

> D. de Laat, N. M. Leijenhorst, W. H. H. de Muinck Keizer,
> *Optimality and uniqueness of the D4 root system*, arXiv:2404.18794,

and deposited by them, under the MIT licence, at 4TU.ResearchData:

    doi:10.4121/74ce1c25-6fca-4680-8a36-e9c18e7e9594
    archive  LasserreSphericalCodes.zip
    MD5      02acd5270f7b3fa799abdeb5291706fd
    size     152065368 bytes (145 MiB) as a zip; unpacked, 331 files
             under proofs/4_24 holding 243787660 bytes (232.5 MiB; on
             disk du -h reports 234M, the "234 megabytes" of the paper)

The paper proves everything it needs about that point itself (the bound, its
equality case, the positivity of the kernel, and the four properties of the
data listed in the proposition "What the verification establishes"); no
statement of the preprint is used.  The data are redistributed here so that
the proof does not depend on a server.

## Contents of this directory

    LasserreSphericalCodes.zip.part-00, -01
                                 the deposited archive, byte for byte, cut into
                                 80 MiB parts because GitHub refuses single files
                                 above 100 MB; `python3 fetch_certificate.py
                                 --join` reassembles LasserreSphericalCodes.zip
                                 and checks its MD5
    LICENSE-MIT                  the licence notice that accompanies the copy
    fetch_certificate.py         downloads the archive from 4TU.ResearchData
                                 and checks its MD5 against the value above
    verify.sh                    unpacks it and runs the seven-step
                                 verification of the paper

## Before tagging a release

Run

    python3 fetch_certificate.py
    python3 fetch_certificate.py --split

in this directory.  The first writes `LasserreSphericalCodes.zip` here and
refuses to keep any file whose checksum differs from the one recorded above;
the second cuts it into parts small enough for the repository.  Commit the
parts (not the whole archive, which `.gitignore` excludes), so that the
snapshot Zenodo archives from the release carries the data the paper says it
carries; the checksum is what fixes the bytes the paper refers to.

## Verifying

    sh verify.sh

unpacks the archive into `proofs/`, runs `multi_cap/llm24_certificate_check.py`
(steps 1, 2, 4, 6 and 7 of the verification: reading, positive definiteness of
the 127 blocks in ball arithmetic and exact LDL^T, the prefactors, the
objective 24, the zeros of the two-point polynomial), then the programs of
`zonal/` (steps 3 and 5: the zonal matrices and the four constraint identities,
in exact rational arithmetic), and finally `lean D4InnerProducts.lean`.  The
run logs of the runs reported in the paper are in `multi_cap/runs/` and
`zonal/runs/`.

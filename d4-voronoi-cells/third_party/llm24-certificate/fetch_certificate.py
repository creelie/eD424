#!/usr/bin/env python3
"""
fetch_certificate.py -- download LasserreSphericalCodes.zip from
4TU.ResearchData and check its MD5 against the value the paper records.

The archive is the data set doi:10.4121/74ce1c25-6fca-4680-8a36-e9c18e7e9594.
The DOI landing page lists the file; this script asks the 4TU (Figshare)
API for the download link of the file of that name, streams it to disk, and
deletes it again if the checksum is wrong.

    python3 fetch_certificate.py            # writes LasserreSphericalCodes.zip here
    python3 fetch_certificate.py --check    # only checks an archive already present
    python3 fetch_certificate.py --split    # after fetching: cut the archive into
                                            # 80 MB parts for the git repository
    python3 fetch_certificate.py --join     # reassemble the parts and check the MD5

GitHub refuses single files above 100 MB, so the archive travels in the
repository as LasserreSphericalCodes.zip.part-00, -01, ...; joining them
gives the deposited archive byte for byte, which the MD5 confirms.
"""
import hashlib
import json
import os
import sys
import urllib.request

DOI = "10.4121/74ce1c25-6fca-4680-8a36-e9c18e7e9594"
NAME = "LasserreSphericalCodes.zip"
MD5 = "02acd5270f7b3fa799abdeb5291706fd"
API = "https://data.4tu.nl/v2/articles?doi=" + DOI
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, NAME)


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check():
    if not os.path.exists(OUT):
        print("missing:", OUT)
        return False
    got = md5_of(OUT)
    ok = got == MD5
    print("MD5 %s  %s  (%s)" % (got, "matches" if ok else "DOES NOT MATCH " + MD5, NAME))
    return ok


def download():
    with urllib.request.urlopen(API, timeout=60) as r:
        articles = json.load(r)
    if not articles:
        sys.exit("the 4TU API returned no record for " + DOI)
    url_files = articles[0]["url_public_api"] + "/files"
    with urllib.request.urlopen(url_files, timeout=60) as r:
        files = json.load(r)
    match = [f for f in files if f.get("name") == NAME]
    if not match:
        sys.exit("no file named %s in the record; files: %s" % (NAME, [f.get("name") for f in files]))
    link = match[0]["download_url"]
    print("downloading", link)
    with urllib.request.urlopen(link, timeout=600) as r, open(OUT, "wb") as f:
        done = 0
        for chunk in iter(lambda: r.read(1 << 20), b""):
            f.write(chunk)
            done += len(chunk)
            if done % (20 << 20) < (1 << 20):
                print("  %d MB" % (done >> 20))
    if not check():
        os.remove(OUT)
        sys.exit("checksum mismatch; the file was removed")


PART = 80 << 20


def split():
    if not check():
        sys.exit(1)
    with open(OUT, "rb") as f:
        k = 0
        while True:
            chunk = f.read(PART)
            if not chunk:
                break
            with open(OUT + ".part-%02d" % k, "wb") as g:
                g.write(chunk)
            k += 1
    print("wrote %d parts" % k)


def join():
    parts = sorted(p for p in os.listdir(HERE) if p.startswith(NAME + ".part-"))
    if not parts:
        sys.exit("no parts found beside this script")
    with open(OUT, "wb") as g:
        for p in parts:
            with open(os.path.join(HERE, p), "rb") as f:
                g.write(f.read())
    print("joined %d parts" % len(parts))
    if not check():
        sys.exit(1)


if __name__ == "__main__":
    if "--split" in sys.argv:
        split(); sys.exit(0)
    if "--join" in sys.argv:
        join(); sys.exit(0)
    if "--check" in sys.argv:
        sys.exit(0 if check() else 1)
    if os.path.exists(OUT) and check():
        print("already present and correct")
    else:
        download()

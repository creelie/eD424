#!/usr/bin/env python3
"""
check_bibliography.py -- checks of the bibliography of paper/D4.tex.

Offline (always):
  * every \\bibitem is cited at least once in the text, and every key that is
    cited has a \\bibitem; no key is defined twice;
  * every DOI link is well formed, and the DOI in the link target equals the
    DOI printed beside it;
  * no DOI occurs in two entries.

Online (with --online, where doi.org, api.crossref.org and api.datacite.org
are reachable): every DOI is looked up, at Crossref or, for DataCite DOIs
(Zenodo, arXiv, Preprints.org), at DataCite, and the registered title and
year are compared with the entry: the script reports the share of the
registered title's words that occur in the entry, and whether the registered
year occurs in it.  A mismatch is printed for a person to judge; the script
does not edit anything.

    python3 check_bibliography.py [path/to/D4.tex] [--online]

Exit status 1 if an offline check fails (or, with --online, if a DOI does not
resolve).
"""
import json
import os
import re
import sys
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
args = [a for a in sys.argv[1:] if not a.startswith('--')]
ONLINE = '--online' in sys.argv
TEX = args[0] if args else os.path.join(HERE, '..', 'paper', 'D4.tex')
src = open(TEX, encoding='utf-8').read()

# strip comments (a % not preceded by a backslash, to the end of the line)
body = re.sub(r'(?<!\\)%.*', '', src)
i0 = body.index(r'\begin{thebibliography}')
i1 = body.index(r'\end{thebibliography}')
text, bib = body[:i0] + body[i1:], body[i0:i1]

# bibitems: \bibitem[label]{key}
items = re.findall(r'\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}', bib)
entries = re.split(r'\\bibitem(?:\[[^\]]*\])?\{[^}]+\}', bib)[1:]
entry_of = dict(zip(items, entries))
cites = []
for m in re.finditer(r'\\(?:cite|citet|citep|nocite)(?:\[[^\]]*\])*\{([^}]+)\}', text):
    cites += [k.strip() for k in m.group(1).split(',')]
fails = []

dups = sorted({k for k in items if items.count(k) > 1})
uncited = [k for k in items if k not in cites]
undefined = sorted({k for k in cites if k not in entry_of})
print('bibitems: %d   distinct keys cited: %d   citations: %d' % (len(items), len(set(cites)), len(cites)))
for name, lst in (('defined twice', dups), ('never cited', uncited), ('cited but not defined', undefined)):
    print('  %-22s %s' % (name + ':', ', '.join(lst) if lst else 'none'))
    if lst:
        fails.append(name)

DOI_RE = re.compile(r'^10\.\d{4,9}/\S+$')
dois = {}
for k, e in entry_of.items():
    for target, shown in re.findall(r'\\bluelink\{https?://(?:dx\.)?doi\.org/([^}]+)\}\{(?:doi:)?([^}]+)\}', e):
        t, s = target.strip(), shown.strip().replace(r'\_', '_')
        t = urllib.parse.unquote(t)
        ok = DOI_RE.match(t) is not None and t.lower() == s.lower()
        if not ok:
            fails.append('doi ' + k)
            print('  DOI MISMATCH in %s: link %s, printed %s' % (k, t, s))
        dois.setdefault(t.lower(), []).append(k)
multi = {d: ks for d, ks in dois.items() if len(ks) > 1}
print('DOIs: %d, each link equal to the printed DOI: %s; shared between entries: %s'
      % (len(dois), 'yes' if not any(f.startswith('doi ') for f in fails) else 'NO',
         ', '.join('%s (%s)' % (d, ' '.join(ks)) for d, ks in multi.items()) or 'none'))
if multi:
    fails.append('shared DOI')

if ONLINE:
    print('\nonline comparison with the registered metadata')
    DATACITE = ('10.5281/', '10.48550/', '10.20944/')

    def get(url):
        req = urllib.request.Request(url, headers={'User-Agent': 'd4-bibliography-check/1.0 (mailto:itsdeep@live.com)'})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)

    def words(s):
        return [w for w in re.findall(r'[a-z0-9]+', s.lower()) if len(w) > 2]

    for d, ks in sorted(dois.items()):
        e = ' '.join(entry_of[k] for k in ks)
        try:
            if d.startswith(DATACITE):
                a = get('https://api.datacite.org/dois/' + urllib.parse.quote(d, safe='/'))['data']['attributes']
                title = a['titles'][0]['title']
                year = str(a.get('publicationYear', ''))
            else:
                m = get('https://api.crossref.org/works/' + urllib.parse.quote(d, safe='/'))['message']
                title = (m.get('title') or [''])[0]
                parts = (m.get('issued') or {}).get('date-parts') or [[None]]
                year = str(parts[0][0] or '')
        except Exception as ex:
            print('  %-40s %-12s DOES NOT RESOLVE (%s)' % (d, ','.join(ks), ex.__class__.__name__))
            fails.append('resolve ' + d)
            continue
        tw = words(title)
        ew = set(words(re.sub(r'\\[a-zA-Z]+', ' ', e)))
        share = sum(w in ew for w in tw) / max(1, len(tw))
        ystat = 'year ok' if year and year in e else 'YEAR %s NOT IN ENTRY' % year
        flag = '' if share >= 0.8 and 'ok' in ystat else '   <-- check'
        print('  %-40s %-10s title words %3.0f%%  %s%s' % (d, ','.join(ks), 100 * share, ystat, flag))

print()
print('FAILED: ' + ', '.join(fails) if fails else 'all bibliography checks passed')
sys.exit(1 if fails else 0)

"""List the decimal numbers of the paper with at least six significant digits
and report which of them occur in no log and in no script under the given
directories (default: the repository this file sits in).

    python3 numscan.py [dir ...]
"""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, '..', 'paper', 'D4.tex')
ROOTS = sys.argv[1:] or [os.path.join(HERE, '..')]
t = open(TEX).read()
t = re.sub(r'(?m)^%.*$', '', t)
nums = {}
for m in re.finditer(r'(?<![\w.])(\d+\.\d{5,})', t):
    s = m.group(1)
    line = t.count('\n', 0, m.start()) + 1
    nums.setdefault(s, line)
corpus = []
for root in ROOTS:
    for dp, dn, fn in os.walk(root):
        if '.git' in dp or '.lake' in dp or 'proofs' in dp:
            continue
        for f in fn:
            if f.endswith(('.log', '.txt', '.py', '.out', '.c', '.lean', '.md')) and os.path.getsize(os.path.join(dp, f)) < 5e7:
                try:
                    corpus.append(open(os.path.join(dp, f), errors='replace').read())
                except Exception:
                    pass
blob = '\n'.join(corpus)
missing = []
for s, line in sorted(nums.items(), key=lambda x: x[1]):
    # accept a match on the number truncated to its first 6 significant digits
    digits = s.replace('.', '').lstrip('0')
    key = s[:max(s.index('.') + 1, len(s) - (len(digits) - 6))] if len(digits) > 6 else s
    if s in blob or key in blob:
        continue
    missing.append((line, s))
print('%d numbers with >= 6 significant digits; %d found nowhere in logs or code' % (len(nums), len(missing)))
for line, s in missing:
    ctx = t.split('\n')[line - 1].strip()[:110]
    print('  line %5d  %-18s %s' % (line, s, ctx))

#!/usr/bin/env python3
"""
gen_count_data.py -- writes CountData<M>.lean from a certificate of
multi_cap/radial_count_sdp.py (multi_cap/radial_certificates/radial_<M>.json),
the certificate of thm:count31 for M = 31.  Every entry is written as the exact
rational that the JSON file holds; nothing is rounded.

Usage: python3 gen_count_data.py ../../multi_cap/radial_certificates/radial_31.json
"""
import json
import sys
from fractions import Fraction as Fr


def rat(x):
    q = Fr(x)
    if q.denominator == 1:
        return '(%d : Rat)' % q.numerator
    return '((%d : Int) : Rat) / %d' % (q.numerator, q.denominator)


def main():
    c = json.load(open(sys.argv[1]))
    M, D, r = c['M'], c['D'], c['r']
    out = []
    out.append('/-')
    out.append('CountData%d.lean: the certificate of the count bound at M = %d (thm:count31 for M = 31),' % (M, M))
    out.append('written by gen_count_data.py from multi_cap/radial_certificates/radial_%d.json.' % M)
    out.append('Every entry is the exact rational of that file.')
    out.append('-/')
    out.append('import CountDomain')
    out.append('')
    out.append('def cert%d : CountCert where' % M)
    out.append('  M := %d' % M)
    out.append('  D := %d' % D)
    out.append('  r := %d' % r)
    out.append('  c1 := %s' % rat(c['c1']))
    out.append('  c2 := %s' % rat(c['c2']))
    out.append('  dmax := %s' % rat(c['dmax']))
    out.append('  m := %s' % rat(c['m']))
    out.append('  z := #[%s]' % ', '.join(rat(x) for x in c['z']))
    mats = []
    for a in c['A']:
        rows = ['#[%s]' % ', '.join(rat(x) for x in row) for row in a]
        mats.append('    #[' + ',\n      '.join(rows) + ']')
    out.append('  A := #[\n' + ',\n'.join(mats) + ']')
    out.append('')
    open('CountData%d.lean' % M, 'w').write('\n'.join(out) + '\n')
    print('wrote CountData%d.lean' % M)


if __name__ == '__main__':
    main()

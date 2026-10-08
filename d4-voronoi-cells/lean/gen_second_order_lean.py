#!/usr/bin/env python3
"""
gen_second_order_lean.py -- write D4SecondOrder.lean from the certificate of
prop:cone-min (multi_cap/second_order/exact_certificate.pkl).

Only the certificate itself is data: the 43 orbits of pairs of cone rows and
the nonnegative rational value of N on each.  Everything else -- the root
system, the vertices of the 24-cell, the tangent bases, the tight pairs and
their triangles, the form H of prop:hessian-d4, the 120 rows of the cone and
the vector c -- is computed inside Lean from the definitions in the file.

Usage: python3 gen_second_order_lean.py   (from this directory)
"""
import os
import pickle

here = os.path.dirname(os.path.abspath(__file__))
cert = pickle.load(open(os.path.join(here, '..', 'multi_cap', 'second_order', 'exact_certificate.pkl'), 'rb'))
orbits, nu = cert['orbits'], cert['nu']

def rat(a, b):
    return '((%d : Rat) / %d)' % (a, b) if b != 1 else '(%d : Rat)' % a

data_orbits = 'def orbits : List (List (Nat × Nat)) := [\n' + ',\n'.join(
    '  [' + ', '.join('(%d, %d)' % (a, b) for a, b in orb) + ']' for orb in orbits) + ']\n'
data_nu = 'def nu : List Rat := [' + ', '.join(rat(a, b) for a, b in nu) + ']\n'

body = open(os.path.join(here, 'D4SecondOrder.template.lean')).read()
out = body.replace('--DATA--', data_orbits + '\n' + data_nu)
open(os.path.join(here, 'D4SecondOrder.lean'), 'w').write(out)
print('wrote D4SecondOrder.lean:', len(orbits), 'orbits,', sum(len(o) for o in orbits), 'pairs')

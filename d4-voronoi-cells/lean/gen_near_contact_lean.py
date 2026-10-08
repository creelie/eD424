#!/usr/bin/env python3
"""
gen_near_contact_lean.py -- writes D4NearContact.lean, the exact arithmetic of
the explicit constant epsilon_0 of thm:near-contact (multi_cap/explicit_eps0.py,
parts B, C, D and the assembly E), from multi_cap/llm24_out/llm24_p2.txt, the
exact coefficients of the two-point polynomial p_2 of the certificate of de
Laat, Leijenhorst and de Muinck Keizer.  The only numbers the Lean file takes
from elsewhere are the two bounds B_3 <= 8.75e7 and B_4 <= 7.34e9 of part A,
which explicit_eps0.py evaluates in ball arithmetic from the deposited
certificate (it prints 8.7499e7 and 7.3316e9).

Usage: python3 gen_near_contact_lean.py   (from the lean/ directory)
"""
import os, sys, math
from fractions import Fraction
sys.set_int_max_str_digits(0)
HERE = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(HERE, '..', 'multi_cap', 'llm24_out', 'llm24_p2.txt')
co = [Fraction(l.split()[1]) for l in open(src) if not l.startswith('#') and l.strip()]
D = 1
for c in co:
    D = D * c.denominator // math.gcd(D, c.denominator)
nums = [c * D for c in co]
assert all(x.denominator == 1 for x in nums)
nums = [int(x) for x in nums]
tmpl = open(os.path.join(HERE, 'D4NearContact.template.lean')).read()
out = tmpl.replace('@@P2NUM@@', ",\n  ".join(str(c) for c in nums)).replace('@@P2DEN@@', str(D))
open(os.path.join(HERE, 'D4NearContact.lean'), 'w').write(out)
print("wrote D4NearContact.lean:", len(nums), "coefficients over a denominator of", len(str(D)), "digits")

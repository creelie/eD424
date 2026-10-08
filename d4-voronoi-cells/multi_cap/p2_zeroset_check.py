#!/usr/bin/env python3
"""
p2_zeroset_check.py

An independent check of the analytic half of thm:m24 of the paper (the
classification of twenty-four contacts).

Input is the two-point polynomial p_2 of the de Laat-Leijenhorst-de Muinck
Keizer certificate, as exact rationals, one coefficient per line in the format

    <index> <numerator>/<denominator>

which is what the deposited data yield.  The script confirms, in exact
rational arithmetic and with no floating point anywhere, that

    deg p_2 = 16,
    p_2(u) = (u+1) (u + 1/2)^2 u^2 (u - 1/2) q(u)  with deg q = 10,
    the four multiplicities are exactly 1, 2, 2, 1,
    q has no real zero in (-1, 1/2]        (Sturm sequence over Q),
    q < 0 on [-1, 1/2],

and therefore that p_2 >= 0 on [-1, 1/2] with zero set exactly
{-1, -1/2, 0, 1/2}.  That is the statement the complementary-slackness step
consumes.

The arithmetic is Python's own fractions module, so the only trust placed
anywhere is in integer arithmetic.  Nothing here shares code with the
authors of the certificate or with the verification in the D4 manuscript.

Usage:  python3 p2_zeroset_check.py [llm24_out/llm24_p2.txt]
"""
import math
import os
import sys
from fractions import Fraction

sys.set_int_max_str_digits(2_000_000)

ROOTS = [(Fraction(-1), 1), (Fraction(-1, 2), 2), (Fraction(0), 2), (Fraction(1, 2), 1)]
LEFT, RIGHT = Fraction(-1), Fraction(1, 2)


def read_poly(path):
    """coefficients c_i of sum_i c_i u^i, as a list indexed from 0"""
    coeff = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        idx, val = line.split(" ", 1)
        coeff[int(idx)] = Fraction(val)
    return [coeff.get(i, Fraction(0)) for i in range(max(coeff) + 1)]


def trim(p):
    while len(p) > 1 and p[-1] == 0:
        p = p[:-1]
    return p


def primitive(p):
    """clear denominators and the integer content; the sign is preserved"""
    lcm = 1
    for c in p:
        lcm = lcm * c.denominator // math.gcd(lcm, c.denominator)
    ints = [int(c * lcm) for c in p]
    g = 0
    for c in ints:
        g = math.gcd(g, abs(c))
    return [Fraction(c // g) for c in ints] if g else [Fraction(0)]


def horner(p, x):
    acc = Fraction(0)
    for c in reversed(p):
        acc = acc * x + c
    return acc


def divide_by_root(p, r):
    """synthetic division by (u - r); returns quotient and remainder"""
    out, acc = [], Fraction(0)
    for c in reversed(p):
        acc = acc * r + c
        out.append(acc)
    return list(reversed(out[:-1])), out[-1]


def derivative(p):
    return trim([p[i] * i for i in range(1, len(p))]) if len(p) > 1 else [Fraction(0)]


def neg_remainder(a, b):
    a, b = list(a), list(b)
    while len(a) >= len(b) and any(c != 0 for c in a):
        if a[-1] == 0:
            a = a[:-1]
            continue
        scale, shift = a[-1] / b[-1], len(a) - len(b)
        for i in range(len(b)):
            a[i + shift] -= scale * b[i]
        a = trim(a)
        if len(a) < len(b):
            break
    return primitive([-c for c in trim(a)])


def sturm_chain(p):
    chain = [primitive(p), primitive(derivative(p))]
    while True:
        nxt = neg_remainder(chain[-2], chain[-1])
        if len(trim(nxt)) == 1 and nxt[0] == 0:
            break
        chain.append(nxt)
        if len(trim(nxt)) == 1:
            break
    return chain


def sign_changes(chain, x):
    vals = [v for v in (horner(p, x) for p in chain) if v != 0]
    return sum(1 for i in range(len(vals) - 1) if (vals[i] > 0) != (vals[i + 1] > 0))


def main(path):
    p = read_poly(path)
    deg = len(trim(p)) - 1
    ok = True

    def report(label, passed, detail=""):
        nonlocal ok
        ok = ok and passed
        print(f"[{'ok  ' if passed else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))

    report("degree of p_2 is 16", deg == 16, f"{deg}")

    cur = trim(p)
    for r, mult in ROOTS:
        for _ in range(mult):
            cur, rem = divide_by_root(cur, r)
            if rem != 0:
                report(f"p_2 divisible by (u - {r}) to order {mult}", False, "remainder nonzero")
                return 1
        _, rem_next = divide_by_root(cur, r)
        report(f"multiplicity at u = {r} is exactly {mult}", rem_next != 0)

    q = trim(cur)
    report("quotient has degree 10", len(q) - 1 == 10, f"{len(q) - 1}")

    chain = sturm_chain(q)
    left, right = sign_changes(chain, LEFT), sign_changes(chain, RIGHT)
    report("q has no real zero in (-1, 1/2]", left - right == 0,
           f"V(-1) = {left}, V(1/2) = {right}")

    probes = [LEFT, Fraction(-9, 10), Fraction(-3, 4), Fraction(-1, 2), Fraction(-1, 4),
              Fraction(0), Fraction(1, 10), Fraction(2, 5), RIGHT]
    signs = {"+" if horner(q, x) > 0 else "-" if horner(q, x) < 0 else "0" for x in probes}
    report("q is strictly negative throughout [-1, 1/2]", signs == {"-"}, "".join(sorted(signs)))

    # p_2 = (u+1)(u+1/2)^2 u^2 (u-1/2) q, so on the interval the sign is
    # (+)(>=0)(>=0)(<=0)(<0) >= 0, with equality exactly at the four roots
    report("p_2 >= 0 on [-1, 1/2], vanishing exactly at -1, -1/2, 0, 1/2", ok)

    print("\nzero set of p_2 on [-1, 1/2] is {-1, -1/2, 0, 1/2}" if ok else "\nCHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm24_out", "llm24_p2.txt")))

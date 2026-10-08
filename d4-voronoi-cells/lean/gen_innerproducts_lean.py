#!/usr/bin/env python3
"""
gen_innerproducts_lean.py -- writes D4InnerProducts.lean from
multi_cap/llm24_out/llm24_p2.txt, the exact coefficients of the two-point
polynomial p_2 of the certificate of de Laat, Leijenhorst and de Muinck
Keizer as computed by multi_cap/llm24_certificate_check.py.  The
coefficients are scaled by their least common denominator so that p_2
becomes an integer polynomial; the Lean file then checks, by kernel
computation on Rat, that p_2 vanishes at -1, -1/2, 0, 1/2 with
multiplicities 1, 2, 2, 1 and that the quotient has no zero on [-1, 1/2]
(Sturm sequence).

Usage: python3 gen_innerproducts_lean.py   (from the lean/ directory)
"""
import os, sys, math
from fractions import Fraction
sys.set_int_max_str_digits(0)
HERE = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(HERE, '..', 'multi_cap', 'llm24_out', 'llm24_p2.txt')
co = [Fraction(l.split()[1]) for l in open(src) if not l.startswith('#')]
D = 1
for c in co:
    D = D * c.denominator // math.gcd(D, c.denominator)
ints = [c * D for c in co]
assert all(x.denominator == 1 for x in ints)
g = 0
for x in ints:
    g = math.gcd(g, int(x))
ints = [int(x) // g for x in ints]

HEAD = "/-\nD4InnerProducts.lean\n\nThe two-point polynomial p_2 of the certificate of de Laat, Leijenhorst and\nde Muinck Keizer (arXiv:2404.18794, data doi:10.4121/74ce1c25-6fca-4680-8a36-e9c18e7e9594),\nas computed exactly from the published data by multi_cap/llm24_certificate_check.py\nand scaled by the least common denominator of its coefficients (an integer of\n15741 digits) so that every coefficient is an integer.  By complementary\nslackness, the inner products of a 24-point code of minimal angle 60 degrees\nare zeros of p_2 in [-1, 1/2] (step 7 of their verification, Lemma 5.1 of\ntheir paper, thm:m24 of ours).  This file proves by kernel computation\nthat p_2 vanishes at -1, -1/2, 0, 1/2 with multiplicities 1, 2, 2, 1, and that\nthe quotient q of p_2 by (u+1)(2u+1)^2 u^2 (2u-1) has no zero on [-1, 1/2]:\nq(-1) and q(1/2) are nonzero and the Sturm sequence of q has the same number\nof sign changes at -1 as at 1/2.  Sturm's theorem, which turns the last\nstatement into the count of zeros, is not formalised here.\n\nNo `sorry`, no Mathlib; compiles against a bare Lean 4 toolchain.\n-/\n\nset_option maxRecDepth 10000\n\nnamespace D4InnerProducts\n\n/-- Coefficients of the scaled p_2, lowest degree first. -/\n"
TAIL = '\n\n/-- Polynomials over the rationals as coefficient lists, lowest degree first. -/\nabbrev Poly := List Rat\n\ndef eval (p : Poly) (x : Rat) : Rat := p.foldr (fun c acc => c + x * acc) 0\n\ndef trim : Poly → Poly\n  | [] => []\n  | c :: cs =>\n    match trim cs with\n    | [] => if c == 0 then [] else [c]\n    | t => c :: t\n\ndef degree (p : Poly) : Nat := (trim p).length - 1\n\ndef deriv (p : Poly) : Poly :=\n  match p with\n  | [] => []\n  | _ :: cs => (List.range cs.length).zipWith (fun i c => ((i : Nat) + 1 : Rat) * c) cs\n\ndef scale (a : Rat) (p : Poly) : Poly := p.map (a * ·)\n\ndef sub (p q : Poly) : Poly :=\n  match p, q with\n  | [], q => q.map (fun c => -c)\n  | p, [] => p\n  | a :: p, b :: q => (a - b) :: sub p q\n\ndef shift (p : Poly) : Poly := 0 :: p\n\n/-- Remainder of p modulo q (q trimmed, nonzero). -/\ndef polyMod (p q : Poly) : Poly :=\n  let q := trim q\n  let dq := q.length - 1\n  let lq := q.getLast! \n  let rec go (r : Poly) (fuel : Nat) : Poly :=\n    match fuel with\n    | 0 => r\n    | fuel + 1 =>\n      let r := trim r\n      if r.length ≤ dq then r\n      else\n        let lr := r.getLast!\n        let k := r.length - 1 - dq\n        let m := scale (lr / lq) (Nat.repeat shift k q)\n        go (sub r m) fuel\n  go p (p.length + 1)\n\n/-- Division of p by (x - a), assuming p(a) = 0; returns the quotient. -/\ndef divLinear (p : Poly) (a : Rat) : Poly :=\n  -- synthetic division from the top coefficient down\n  let rev := (trim p).reverse\n  let rec go : List Rat → Rat → List Rat\n    | [], _ => []\n    | c :: cs, carry =>\n      let b := c + a * carry\n      b :: go cs b\n  match rev with\n  | [] => []\n  | _ => ((go rev 0).dropLast).reverse\n\ndef sturm (p : Poly) : List Poly :=\n  let rec go (a b : Poly) (fuel : Nat) : List Poly :=\n    match fuel with\n    | 0 => [a]\n    | fuel + 1 =>\n      if trim b == [] then [a]\n      else a :: go b (scale (-1) (polyMod a b)) fuel\n  go (trim p) (trim (deriv p)) (p.length + 1)\n\ndef signOf (x : Rat) : Int := if x > 0 then 1 else if x < 0 then -1 else 0\n\ndef variations (ps : List Poly) (x : Rat) : Nat :=\n  let s := (ps.map (fun p => signOf (eval p x))).filter (· != 0)\n  let rec cnt : List Int → Nat\n    | a :: b :: rest => (if a != b then 1 else 0) + cnt (b :: rest)\n    | _ => 0\n  cnt s\n\ndef p2q : Poly := p2.map (fun c => Rat.ofInt c)\n\ntheorem p2_degree : degree p2q = 16 := by decide +kernel\n\ntheorem p2_vanishes :\n    eval p2q (-1) = 0 ∧ eval p2q (-1/2) = 0 ∧ eval p2q 0 = 0 ∧ eval p2q (1/2) = 0 := by\n  decide +kernel\n\n/-- The quotient after dividing out the four zeros with multiplicities 1, 2, 2, 1. -/\ndef q : Poly :=\n  divLinear (divLinear (divLinear (divLinear (divLinear (divLinear p2q (-1)) (-1/2)) (-1/2)) 0) 0) (1/2)\n\ntheorem q_degree : degree q = 10 := by decide +kernel\n\n/-- The multiplicities are exactly 1, 2, 2, 1: the quotient does not vanish at\n    any of the four points. -/\ntheorem q_nonzero_at_the_four :\n    eval q (-1) ≠ 0 ∧ eval q (-1/2) ≠ 0 ∧ eval q 0 ≠ 0 ∧ eval q (1/2) ≠ 0 := by\n  decide +kernel\n\n/-- p_2 = (u+1)(2u+1)^2 u^2 (2u-1) q up to the scaling by 16: the division is exact. -/\ntheorem division_exact :\n    eval (divLinear p2q (-1)) (-1/2) = 0 ∧\n    eval (divLinear (divLinear p2q (-1)) (-1/2)) (-1/2) = 0 ∧\n    eval (divLinear (divLinear (divLinear p2q (-1)) (-1/2)) (-1/2)) 0 = 0 ∧\n    eval (divLinear (divLinear (divLinear (divLinear p2q (-1)) (-1/2)) (-1/2)) 0) 0 = 0 ∧\n    eval (divLinear (divLinear (divLinear (divLinear (divLinear p2q (-1)) (-1/2)) (-1/2)) 0) 0) (1/2) = 0 := by\n  decide +kernel\n\n/-- The Sturm sequence of q has as many sign changes at -1 as at 1/2, so q has\n    no zero in (-1, 1/2]; together with q(-1) ≠ 0 there is none on [-1, 1/2]. -/\ntheorem sturm_no_zero : variations (sturm q) (-1) = variations (sturm q) (1/2) := by\n  decide +kernel\n\ntheorem sturm_length : (sturm q).length = 11 := by decide +kernel\n\nend D4InnerProducts\n\n#print axioms D4InnerProducts.p2_vanishes\n#print axioms D4InnerProducts.q_nonzero_at_the_four\n#print axioms D4InnerProducts.division_exact\n#print axioms D4InnerProducts.sturm_no_zero\n'
out = HEAD + "def p2 : List Int := [\n" + ",\n".join("  " + str(c) for c in ints) + "]\n" + TAIL
open(os.path.join(HERE, 'D4InnerProducts.lean'), 'w').write(out)
print("wrote D4InnerProducts.lean:", len(ints), "coefficients, least common denominator of", len(str(D)), "digits")
